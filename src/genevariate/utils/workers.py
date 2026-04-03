"""
Worker threads for background processing.
Handles GSE extraction, sample classification, and other threaded tasks.
"""

import threading
import pandas as pd
import sqlite3
import re
from pathlib import Path


class ExtractionThread(threading.Thread):
    """Thread for extracting GSEs from GEOmetadb based on search criteria."""
    
    def __init__(self, plat_filter, search_tokens, log_func, on_finish):
        super().__init__(daemon=True)
        self.plat_filter = plat_filter
        self.search_tokens = search_tokens
        self.log_func = log_func
        self.on_finish = on_finish
        self.final_df = None
        self.gse_keywords = {}
        self.gse_descriptions = {}
        self._stop_event = threading.Event()
    
    def stop(self):
        """Stop the thread."""
        self._stop_event.set()
    
    def run(self):
        """Main extraction logic."""
        try:
            from genevariate.config import CONFIG
            from genevariate.core.db_loader import open_geometadb
            import os

            # Check for stop signal
            if self._stop_event.is_set():
                self.final_df = pd.DataFrame()
                self.on_finish()
                return

            # Open database (resource-aware: disk or RAM depending on device)
            gz_path = str(CONFIG['paths']['geo_db'])
            conn = open_geometadb(gz_path, log_fn=self.log_func)
            if conn is None:
                self.log_func("[Step 1] ERROR: Could not open GEOmetadb")
                self.final_df = pd.DataFrame()
                self.on_finish()
                return

            # Parse search tokens
            tokens = [t.strip().lower() for t in self.search_tokens.split(',') if t.strip()]

            self.log_func(f"[Step 1] Searching for: {', '.join(tokens)}")

            # Build SQL query
            if self.plat_filter:
                plat_list = [p.strip().upper() for p in self.plat_filter.split(',')]
                plat_placeholders = ','.join(['?'] * len(plat_list))

                query = f"""
                    SELECT DISTINCT gse.gse AS series_id, gse.title, gse.summary,
                           gsm.gsm, gsm.title AS gsm_title, gse_gpl.gpl
                    FROM gse
                    JOIN gse_gsm ON gse.gse = gse_gsm.gse
                    JOIN gsm ON gse_gsm.gsm = gsm.gsm
                    JOIN gse_gpl ON gse.gse = gse_gpl.gse
                    WHERE gse_gpl.gpl IN ({plat_placeholders})
                """
                params = plat_list
            else:
                query = """
                    SELECT DISTINCT gse.gse AS series_id, gse.title, gse.summary,
                           gsm.gsm, gsm.title AS gsm_title, gse_gpl.gpl
                    FROM gse
                    JOIN gse_gsm ON gse.gse = gse_gsm.gse
                    JOIN gsm ON gse_gsm.gsm = gsm.gsm
                    JOIN gse_gpl ON gse.gse = gse_gpl.gse
                """
                params = []

            if self._stop_event.is_set():
                conn.close()
                self.final_df = pd.DataFrame()
                self.on_finish()
                return

            self.log_func("[Step 1] Querying database...")
            df = pd.read_sql_query(query, conn, params=params)

            conn.close()
            
            if df.empty:
                self.log_func("[Step 1] No experiments found in database")
                self.final_df = pd.DataFrame()
                self.on_finish()
                return
            
            self.log_func(f"[Step 1] Found {len(df)} total samples, filtering by keywords...")
            
            # Filter by keywords
            def matches_keywords(row):
                text = f"{row.get('title', '')} {row.get('summary', '')} {row.get('gsm_title', '')}".lower()
                return any(token in text for token in tokens)
            
            df['matches'] = df.apply(matches_keywords, axis=1)
            filtered_df = df[df['matches']].copy()
            filtered_df.drop(columns=['matches'], inplace=True)
            
            if filtered_df.empty:
                self.log_func("[Step 1] No experiments matched the keywords")
                self.final_df = pd.DataFrame()
                self.on_finish()
                return
            
            # Extract keyword matches for each GSE
            for gse in filtered_df['series_id'].unique():
                gse_rows = filtered_df[filtered_df['series_id'] == gse]
                first_row = gse_rows.iloc[0]
                
                text = f"{first_row.get('title', '')} {first_row.get('summary', '')}".lower()
                matched = [t for t in tokens if t in text]
                self.gse_keywords[gse] = matched
                self.gse_descriptions[gse] = first_row.get('summary', 'No description')[:200]
            
            self.final_df = filtered_df
            self.log_func(f"[Step 1] ✓ Filtered to {len(filtered_df)} samples across {filtered_df['series_id'].nunique()} experiments")
            
            self.on_finish()
            
        except Exception as e:
            import traceback
            self.log_func(f"[Step 1 ERROR] {traceback.format_exc()}")
            self.final_df = pd.DataFrame()
            self.on_finish()


class LabelingThread(threading.Thread):
    """Thread for AI-powered sample labeling."""
    
    def __init__(self, input_dataframe, ai_agent, gui_log_func, on_finish):
        super().__init__(daemon=True)
        self.input_dataframe = input_dataframe
        self.ai_agent = ai_agent
        self.gui_log_func = gui_log_func
        self.on_finish = on_finish
        self.result_df = None
        self._stop_event = threading.Event()
    
    def stop(self):
        """Stop the thread."""
        self._stop_event.set()
    
    def run(self):
        """Main labeling logic."""
        try:
            self.gui_log_func("[AI] Starting classification process...")
            
            # Use the AI agent to process samples
            self.result_df = self.ai_agent.process_samples(self.input_dataframe)
            
            if self.result_df is not None and not self.result_df.empty:
                self.gui_log_func(f"[AI] ✓ Classified {len(self.result_df)} samples")
            else:
                self.gui_log_func("[AI] ✗ Classification produced no results")
                self.result_df = None
            
            self.on_finish()
            
        except Exception as e:
            import traceback
            self.gui_log_func(f"[AI ERROR] {traceback.format_exc()}")
            self.result_df = None
            self.on_finish()


class SampleClassificationAgent:
    """
    AI agent for classifying biological samples using Ollama.
    Processes samples in parallel batches.
    """
    
    def __init__(self, tools_list, gui_log_func, max_workers=4):
        self.tools = tools_list
        self.log_func = gui_log_func
        self.max_workers = max_workers
    
    def process_samples(self, dataframe):
        """
        Process samples and classify them.
        
        Args:
            dataframe: DataFrame with sample metadata
            
        Returns:
            DataFrame with classification columns added
        """
        if dataframe is None or dataframe.empty:
            return pd.DataFrame()
        
        try:
            from genevariate.core.nlp import classify_sample
            import concurrent.futures
            
            # Ensure GSM column exists
            if 'GSM' not in dataframe.columns:
                if 'gsm' in dataframe.columns:
                    dataframe.rename(columns={'gsm': 'GSM'}, inplace=True)
                else:
                    self.log_func("[AI ERROR] No GSM column found")
                    return pd.DataFrame()
            
            results = []
            total = len(dataframe)
            
            self.log_func(f"[AI] Processing {total} samples with {self.max_workers} workers...")
            
            # Process in batches
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                
                for idx, row in dataframe.iterrows():
                    future = executor.submit(classify_sample, row)
                    futures[future] = idx
                
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        
                        if completed % 10 == 0:
                            self.log_func(f"[AI] Progress: {completed}/{total} samples classified")
                    
                    except Exception as e:
                        self.log_func(f"[AI WARNING] Failed to classify sample: {e}")
                        row_idx = futures[future]
                        results.append({
                            'GSM': dataframe.iloc[row_idx].get('GSM', 'Unknown'),
                            'Classified_Condition': 'Unknown',
                            'Classified_Tissue': 'Unknown',
                            'Classified_Treatment': 'Unknown',
                            'Classified_Age': None,
                            'Classified_Time': None,
                            'Classified_Dosage': None,
                        })
            
            if not results:
                return pd.DataFrame()
            
            result_df = pd.DataFrame(results)
            
            # Merge with original dataframe
            final_df = dataframe.merge(result_df, on='GSM', how='left')
            
            return final_df
            
        except Exception as e:
            import traceback
            self.log_func(f"[AI ERROR] {traceback.format_exc()}")
            return pd.DataFrame()
