import concurrent.futures
import datetime
import logging
from typing import Any

from google.auth import exceptions as auth_exceptions
import pandas as pd
from pydantic import BaseModel

from .config import AdAnalyserConfig
from .exceptions import DataValidationError
from .gemini import GeminiAIManager
from .storage import StorageManager

try:
    from airflow.utils.log.logging_mixin import LoggingMixin
    logger = LoggingMixin().log
except ImportError:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("ad_analyser").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class AdAnalyser:
    def __init__(self, config: AdAnalyserConfig):
        self.config = config
        self.logger = logger
        self.ai_fields = []
        schema = self.config.response_schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if hasattr(schema, 'model_fields'):
                self.ai_fields = list(schema.model_fields.keys())
            else:
                self.ai_fields = list(getattr(self.config.response_schema, '__fields__').keys())
        elif isinstance(schema, dict):
            self.ai_fields = list(schema.get('properties', {}).keys())

        self.storage = StorageManager(
            gcp_project_id=self.config.gcp_project_id,
            gcp_region=self.config.gcp_region,
            gcs_bucket=self.config.gcs_bucket,
            gcs_prefix=self.config.gcs_prefix
        )
        self.ai_manager = GeminiAIManager(
            config=self.config,
            storage_manager=self.storage,
            ai_fields=self.ai_fields
        )

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        mandatory_columns = {'creative_id', 'creative_url_supplier'}
        if not mandatory_columns.issubset(df_copy.columns):
            missing_cols = mandatory_columns - set(df_copy.columns)
            raise DataValidationError(f"Missing mandatory columns: {missing_cols}")

        cols_to_clean = ['advertiser_name', 'creative_landingpage_url']
        for col in cols_to_clean:
            if col in df_copy.columns:
                 df_copy[col] = df_copy[col].fillna("")

        return df_copy

    def _chunkify(self, df: pd.DataFrame, chunk_size: int):
        for start in range(0, len(df), chunk_size):
            yield df[start:start + chunk_size]

    def _process_single_row(self, row: dict[str, Any]) -> dict[str, Any]:
        results_row = row.copy()
        results_row.update({"status": None, "error_message": None, "processed_at": None})
        for field in self.ai_fields:
            results_row[field] = None

        try:
            url = row.get('creative_url_supplier')
            if not url:
                raise ValueError("Missing creative URL")

            prompt_parts = ["Analyze this ad."]
            if row.get('advertiser_name'):
                prompt_parts.append(f"Advertiser Name: {row.get('advertiser_name')}")
            if row.get('creative_landingpage_url'):
                prompt_parts.append(f"Landing Page URL: {row.get('creative_landingpage_url')}")
            if row.get('creative_campaign_name'):
                prompt_parts.append(f"Campaign Name: {row.get('creative_campaign_name')}")
            if row.get('social_headline_text'):
                prompt_parts.append(f"Headline: {row.get('social_headline_text')}")
            if row.get('social_description'):
                prompt_parts.append(f"Description: {row.get('social_description')}")
            if row.get('social_campaign_text'):
                prompt_parts.append(f"Campaign Text: {row.get('social_campaign_text')}")
            prompt = "\n".join(prompt_parts)

            parsed_response = self.ai_manager.analyze_creative(creative_url=url, prompt=prompt)
            update_data = {
                "status": "SUCCESS",
                "error_message": None,
                "processed_at": datetime.datetime.now().isoformat()
            }
            for field in self.ai_fields:
                if isinstance(parsed_response, dict):
                    update_data[field] = parsed_response.get(field)
            results_row.update(update_data)
        
        except Exception as err:
            err_str = str(err).lower()
            if (isinstance(err, auth_exceptions.DefaultCredentialsError) or 
                "service account" in err_str):
                raise err
            self.logger.error(f"Error processing {row.get('creative_id')}: {err}")
            results_row.update({
                "status": "FAILED",
                "error_message": str(err)[:500],
                "processed_at": datetime.datetime.now().isoformat()
            })
        return results_row

    def _process_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_row = {executor.submit(self._process_single_row, row): row for row in records}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_row)):
                try:
                    results.append(future.result())
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if (isinstance(exc, auth_exceptions.DefaultCredentialsError) or 
                        "service account" in exc_str):
                        self.logger.error("Critical error: Service account missing.")
                        raise exc
                    self.logger.error(f"Row generated exception: {exc}")
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Processed {i+1}/{len(records)}")

        if self.config.dest_table:
            self.storage.flush_to_bq(
                dest_table=self.config.dest_table, 
                results=results, 
                include_logs=self.config.include_logs
            )
        return results

    def run_job(self, limit: int | None = None) -> pd.DataFrame | None:
        if self.config.df is not None and not self.config.df.empty:
            return self._run_dataframe_job(limit)
        elif self.config.input_query:
            return self._run_query_job(limit)
        else:
            self.logger.error("No DataFrame or Input Query provided.")
            return None

    def _run_dataframe_job(self, limit: int | None) -> pd.DataFrame | None:
        assert self.config.df is not None, "DataFrame must not be None"
        work_df = self.config.df.copy()
        work_df = self._prepare_dataframe(work_df)
        
        possible_context_cols = ['advertiser_name', 'creative_landingpage_url']
        present_context_cols = [col for col in possible_context_cols if col in work_df.columns]
        dedup_keys = ['creative_url_supplier'] + present_context_cols
        
        try:
            unique_rows_df = work_df.drop_duplicates(subset=dedup_keys)
        except KeyError as e:
            self.logger.error(f"Deduplication failed: {e}.")
            unique_rows_df = work_df.drop_duplicates()
        
        if limit:
            unique_rows_df = unique_rows_df.head(limit)

        all_results = []
        df_source = self._chunkify(unique_rows_df, self.config.chunk_size)
        total_processed = 0
        
        for idx, chunk_df in enumerate(df_source):
            if chunk_df.empty: 
                continue            
            chunk_res = self._process_records(chunk_df.to_dict('records'))
            all_results.extend(chunk_res)
            total_processed += len(chunk_res)

        if all_results:
            results_df = pd.DataFrame(all_results)
            generated_fields = self.ai_fields[:]
            cols_mask_merge = generated_fields + ['status']
            if self.config.include_logs:
                cols_mask_merge += ['error_message', 'processed_at']
            candidate_cols = dict.fromkeys(dedup_keys + cols_mask_merge)
            cols_to_merge = [c for c in candidate_cols if c in results_df.columns]
            
            cols_to_drop = list(set(generated_fields) - set(dedup_keys))
            work_df = work_df.drop(columns=cols_to_drop, errors="ignore")

            merged_df = pd.merge(work_df, results_df[cols_to_merge], how='left', on=dedup_keys)

            if not self.config.include_logs and 'status' in merged_df.columns:
                merged_df = merged_df[merged_df['status'] != 'FAILED']
                merged_df = merged_df.drop(columns=['status'], errors='ignore')
            
            if self.config.output_path:
                self.storage.save_results(
                    output_path=self.config.output_path, 
                    results=merged_df.to_dict('records'), 
                    include_logs=True 
                )
            
            if not self.config.output_path and not self.config.dest_table:
                return merged_df
        return None

    def _run_query_job(self, limit: int | None) -> pd.DataFrame | None:
        total_processed = 0
        all_query_results = []
        
        assert self.config.input_query is not None, "Input query must not be None"
        df_source = self.storage.execute_query(self.config.input_query, self.config.chunk_size)

        for idx, df in enumerate(df_source):
            if df.empty: 
                continue
            
            try:
                df = self._prepare_dataframe(df)
            except DataValidationError as e:
                self.logger.error(f"Validation error: {e}")
                continue

            records = df.to_dict('records')
            if limit:
                remaining = limit - total_processed
                if remaining <= 0: 
                    break
                if len(records) > remaining: 
                    records = records[:remaining]

            if not records: 
                continue

            chunk_res = self._process_records(records)
            
            if not self.config.output_path and not self.config.dest_table:
                all_query_results.extend(chunk_res)
            elif self.config.output_path:
                self.storage.save_results(
                    output_path=self.config.output_path, 
                    results=chunk_res, 
                    batch_id=idx,
                    include_logs=self.config.include_logs
                )
            
            total_processed += len(chunk_res)
            if limit and total_processed >= limit:
                break
            
        if not self.config.output_path and not self.config.dest_table:
            final_df = pd.DataFrame(all_query_results)
            if not self.config.include_logs and not final_df.empty and 'status' in final_df.columns:
                 final_df = final_df[final_df['status'] != 'FAILED']
                 final_df = final_df.drop(columns=['status', 'error_message', 'processed_at'], 
                                          errors='ignore')
            return final_df
        return None
