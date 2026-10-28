import logging
import os
import threading
from typing import Any

from google.cloud import bigquery, storage
import pandas as pd

from .exceptions import IntegrationError

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, 
                 gcp_project_id: str | None = None,
                 gcp_region: str | None = None,
                 gcs_bucket: str | None = None,
                 gcs_prefix: str | None = None):
        
        self.gcp_project_id = gcp_project_id
        self.gcp_region = gcp_region
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        
        self._bq_client = None
        self._gcs_client = None
        self._bq_lock = threading.Lock()
        self._gcs_lock = threading.Lock()

    @property
    def bq_client(self) -> bigquery.Client:
        with self._bq_lock:
            if not self._bq_client:
                self._bq_client = bigquery.Client(project=self.gcp_project_id, 
                                                  location=self.gcp_region)
            return self._bq_client

    @property
    def gcs_client(self) -> storage.Client:
        with self._gcs_lock:
            if not self._gcs_client:
                self._gcs_client = storage.Client(project=self.gcp_project_id)
            return self._gcs_client

    def upload_to_gcs(self, file_name: str, mime_type: str) -> str:
        if not self.gcs_bucket or not self.gcs_prefix:
            raise IntegrationError("GCS bucket or prefix not configured.")
            
        try:
            object_name = f"{self.gcs_prefix}/{os.path.basename(file_name)}"
            bucket = self.gcs_client.bucket(self.gcs_bucket)
            blob = bucket.blob(object_name)
            blob.upload_from_filename(file_name, content_type=mime_type)
            return f"gs://{self.gcs_bucket}/{object_name}"
        except Exception as e:
            raise IntegrationError(f"Failed to upload to GCS: {e}") from e

    def read_bytes_from_gcs(self, gcs_uri: str) -> bytes:
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        parts = gcs_uri[5:].split("/", 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        try:
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.download_as_bytes()
        except Exception as e:
            raise IntegrationError(f"Failed to download from GCS: {e}") from e

    def flush_to_bq(self, 
                    dest_table: str, 
                    results: list[dict[str, Any]], 
                    include_logs: bool = False):
        if not results or not dest_table:
            return

        try:
            if not include_logs:
                valid_results = [row for row in results if row.get('status') != 'FAILED']
                log_cols = {'status', 'error_message', 'processed_at'}
                rows_to_insert = [
                    {k: v for k, v in row.items() if k not in log_cols}
                    for row in valid_results
                ]
            else:
                rows_to_insert = results

            if not rows_to_insert:
                return

            errors = self.bq_client.insert_rows_json(dest_table, rows_to_insert)
            if not errors:
                logger.info(f"Flushed {len(rows_to_insert)} rows to BigQuery -> {dest_table}")
            else:
                raise IntegrationError(f"BigQuery Insertion Errors: {errors}")
        except Exception as e:
            raise IntegrationError(f"BigQuery Flush Failure: {e}") from e

    def save_results(self, 
                     output_path: str, 
                     results: list[dict[str, Any]], 
                     batch_id: int | None = None, 
                     include_logs: bool = False):
        if not results or not output_path:
            return

        try:
            if not include_logs:
                valid_results = [row for row in results if row.get('status') != 'FAILED']
                log_cols = {'status', 'error_message', 'processed_at'}
                filtered_results = [
                    {k: v for k, v in row.items() if k not in log_cols}
                    for row in valid_results
                ]
                df_results = pd.DataFrame(filtered_results)
            else:
                df_results = pd.DataFrame(results)  
            
            if df_results.empty:
                return

            base_name, ext = os.path.splitext(output_path)
            if batch_id is not None:
                final_path = f"{base_name}_part_{batch_id:05d}{ext}"
            else:
                final_path = output_path

            if output_path.lower().startswith("gs://"):
                parts = final_path[5:].split('/', 1)
                if len(parts) < 2:
                    raise IntegrationError(f"Invalid GCS URI: {final_path}")
                    
                bucket_name, object_name = parts[0], parts[1]
                
                if ext == '.xlsx':
                    object_name = object_name.replace('.xlsx', '.csv')
                     
                csv_data = df_results.to_csv(index=False)
                bucket = self.gcs_client.bucket(bucket_name)
                blob = bucket.blob(object_name)
                blob.upload_from_string(csv_data, content_type='text/csv')
            else:
                parent_dir = os.path.dirname(final_path)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

                if ext == '.xlsx':
                    df_results.to_excel(final_path, index=False)
                else:
                    file_exists = os.path.exists(output_path)
                    df_results.to_csv(output_path, 
                                      index=False, 
                                      mode='a' if file_exists else 'w', 
                                      header=not file_exists)

        except Exception as e:
            raise IntegrationError(f"Failed to save results: {e}") from e

    def execute_query(self, query: str, chunk_size: int):
         try:
             query_job = self.bq_client.query(query)
             return query_job.result(page_size=chunk_size).to_dataframe_iterable()
         except Exception as e:
             raise IntegrationError(f"Query execution failed: {e}") from e
