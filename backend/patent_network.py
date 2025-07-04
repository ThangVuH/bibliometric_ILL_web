"""
Patent Data Pipeline
A clean, modular system for scraping patent data from Lens.org and analyzing it.

Author: Refactored from original draft_patent.py
"""

import os
import sys
import time
import random
import json
import glob
import logging
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

# Web scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# Database
from sqlalchemy import create_engine, Table, Column, Integer, Date, String, MetaData, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('patent_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the patent pipeline."""
    
    # def __init__(self, config_dict: Optional[Dict] = None):
    def __init__(self, config_dict = None):
        if config_dict is None:
            config_dict = self._get_default_config()
        
        self.URL = config_dict.get("URL", "https://www.lens.org/lens/patcite")
        self.download_folder = config_dict.get("download_folder", "./downloads")
        self.input_file_path = config_dict.get("input_file_path", "./list_doi.txt")
        self.db_connection_string = config_dict.get("db_connection_string", 
                                                   "postgresql://postgres:1234@localhost:5432/test_corpus")
        self.output_folder = config_dict.get("output_folder", "./output")
        
        # Create directories if they don't exist
        Path(self.download_folder).mkdir(parents=True, exist_ok=True)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _get_default_config() -> Dict:
        return {
            "URL": "https://www.lens.org/lens/patcite",
            "download_folder": "./downloads",
            "input_file_path": "./list_doi.txt",
            "db_connection_string": "postgresql://postgres:1234@localhost:5432/test_corpus",
            "output_folder": "./output"
        }


class FileManager:
    """Handles file operations for downloads and data processing."""
    
    @staticmethod
    def get_most_recent_file(download_folder: str, pattern: str = "*.xls*") -> Optional[str]:
        """Get the most recently created file matching the pattern."""
        try:
            list_of_files = glob.glob(os.path.join(download_folder, pattern))
            if not list_of_files:
                return None
            latest_file = max(list_of_files, key=os.path.getctime)
            return latest_file
        except Exception as e:
            logger.error(f"Error finding recent file: {e}")
            return None
    
    @staticmethod
    def rename_file(old_path: str, new_name: str, download_folder: str) -> str:
        """Rename a file to a new name, preserving extension."""
        try:
            file_extension = os.path.splitext(old_path)[1]
            new_file_path = os.path.join(download_folder, new_name + file_extension)
            
            if os.path.exists(new_file_path):
                # If file exists, add timestamp
                timestamp = int(time.time())
                new_file_path = os.path.join(download_folder, f"{new_name}_{timestamp}{file_extension}")
            
            os.rename(old_path, new_file_path)
            logger.info(f"File renamed to: {new_file_path}")
            return new_file_path
        except Exception as e:
            logger.error(f"Error renaming file: {e}")
            raise
    
    @staticmethod
    def clean_and_rename_download(download_folder: str, new_file_name: str) -> str:
        """Find the most recent download and rename it."""
        downloaded_file = FileManager.get_most_recent_file(download_folder)
        if not downloaded_file:
            raise FileNotFoundError("No Excel file found in the download folder.")
        
        logger.info(f"File downloaded successfully: {downloaded_file}")
        
        new_file = f"{new_file_name}.xlsx"
        expected_path = os.path.join(download_folder, new_file)
        
        if downloaded_file != expected_path:
            new_file_path = FileManager.rename_file(downloaded_file, new_file_name, download_folder)
        else:
            new_file_path = expected_path
        
        return new_file_path


class WebScraper:
    """Handles web scraping from Lens.org."""
    
    def __init__(self, config: Config):
        self.config = config
        self.driver = None
    
    def _setup_driver(self) -> webdriver.Firefox:
        """Set up Firefox WebDriver with download preferences."""
        options = Options()
        options.set_preference("browser.download.folderList", 2)
        options.set_preference("browser.download.dir", self.config.download_folder)
        options.set_preference("browser.helperApps.neverAsk.saveToDisk", 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        options.add_argument("--headless")  # Optional: run in headless mode
        
        return webdriver.Firefox(options=options)
    
    def _click_element(self, by: By, value: str, timeout: int = 20) -> None:
        """Helper function to click an element with WebDriverWait."""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            element.click()
            self._random_sleep(2, 6)
        except TimeoutException as e:
            logger.error(f"Timeout waiting for element {value}: {e}")
            raise
    
    @staticmethod
    def _random_sleep(min_sec: int = 2, max_sec: int = 6) -> None:
        """Sleep for a random amount of time to avoid being detected as a bot."""
        time.sleep(random.randint(min_sec, max_sec))
    
    def scrape_patent_data(self, file_path: str) -> Tuple[str, str]:
        """
        Scrape patent data from Lens.org using the provided DOI file.
        Returns paths to the two downloaded Excel files.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        self.driver = self._setup_driver()
        
        try:
            # Navigate to Lens.org
            logger.info(f"Navigating to {self.config.URL}")
            self.driver.get(self.config.URL)
            self._random_sleep(2, 6)
            
            # Handle cookie notice
            self._click_element(By.XPATH, "//div[@class='kill-cookie-notice']//a[@class='btn btn-xs btn-feat ng-binding']")
            self._click_element(By.XPATH, "//a[@class='btn btn-sm btn-danger-txt ng-binding']")
            
            # Switch to iframe
            iframe = self.driver.find_element(By.TAG_NAME, "iframe")
            self.driver.switch_to.frame(iframe)
            self._random_sleep(2, 6)
            
            # Upload file
            logger.info(f"Uploading file: {file_path}")
            file_input = self.driver.find_element(By.CSS_SELECTOR, "input[type='file']")
            self.driver.execute_script("arguments[0].style.display = 'block';", file_input)
            file_input.send_keys(file_path)
            self._random_sleep(2, 6)
            
            # Submit
            submit_button = self.driver.find_element(By.CLASS_NAME, "Button")
            submit_button.click()
            self._random_sleep(2, 6)
            
            return self._process_search_results()
            
        except Exception as e:
            logger.error(f"Error during web scraping: {e}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
    
    def _process_search_results(self) -> Tuple[str, str]:
        """Process search results and download Excel files."""
        try:
            # Check if search results exist
            result_header = self.driver.find_element(
                By.XPATH, "//div[@class='Header']/h1[contains(text(),'Search Results')]"
            )
            logger.info("Search results found! Proceeding to download...")
            
            retry_button = self.driver.find_element(By.XPATH, "//button[@class='Button']")
            retry_button.click()
            self._random_sleep(2, 6)
            
            # Download first Excel file (patent documents)
            logger.info("Downloading first Excel file (patent documents)...")
            export_patent = self.driver.find_element(By.CLASS_NAME, "Select-placeholder")
            export_patent.click()
            self._random_sleep(2, 6)
            
            export_excel = self.driver.find_element(
                By.XPATH, "//div[@class='Select-menu-outer']//div[@class='Select-menu']//div[@class='Select-option is-focused']"
            )
            export_excel.click()
            self._random_sleep(30, 35)
            
            patent_file = FileManager.clean_and_rename_download(self.config.download_folder, 'patent_doc')
            self._random_sleep(4, 8)
            
            # Download second Excel file (citations)
            logger.info("Downloading second Excel file (citations)...")
            export_patent = self.driver.find_element(By.ID, "react-select-7--value")
            export_patent.click()
            self._random_sleep(2, 6)
            
            export_excel = self.driver.find_element(By.ID, "react-select-7--option-0")
            export_excel.click()
            self._random_sleep(30, 35)
            
            citation_file = FileManager.clean_and_rename_download(self.config.download_folder, 'cit_article')
            
            logger.info("Both files downloaded successfully")
            return patent_file, citation_file
            
        except NoSuchElementException:
            logger.warning("No results found, attempting to start new search...")
            try:
                retry_button = self.driver.find_element(By.XPATH, "//button[@class='Button']")
                retry_button.click()
                logger.info("New search started.")
                raise ValueError("No search results found for the provided DOIs")
            except NoSuchElementException:
                logger.error("Fallback button not found.")
                raise


class DataProcessor:
    """Handles data preprocessing and cleaning."""
    
    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess a DataFrame."""
        df = df.copy()
        
        # Clean column names (snake_case, no special characters)
        cleaned_columns = []
        for col in df.columns:
            col_clean = col.strip().lower()
            # col_clean = col_clean.replace("(", "").replace(")", "")  # Remove parentheses
            # col_clean = ''.join(c if c.isalnum() else '_' for c in col_clean)  # Replace non-alphanumeric with underscore
            # col_clean = '_'.join(filter(None, col_clean.split('_')))  # Remove multiple underscores
            
            col_clean = re.sub(r"\(.*?\)", "", col_clean)            # remove anything in parentheses
            col_clean = re.sub(r"[^a-z0-9]+", "_", col_clean)         # replace non-alphanumeric with underscores
            col_clean = re.sub(r"_+", "_", col_clean).strip("_")      # remove redundant underscores
        
            cleaned_columns.append(col_clean)
        
        df.columns = cleaned_columns
        
        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Convert numeric columns
        year_cols = [col for col in df.columns if any(keyword in col for keyword in ['year', 'count', 'size'])]
        for col in year_cols:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype('Int64')  # Handles NaNs
        
        return df
    
    @staticmethod
    def process_citation_data(df: pd.DataFrame) -> pd.DataFrame:
        """Process citation-specific data."""
        df = df.copy()
        
        # Extract DOI from citation_external_id
        if 'citation_external_id' in df.columns:
            df['doi'] = df['citation_external_id'].apply(
                lambda x: x.split('\n')[0].split(':')[1] if pd.notna(x) and ':' in x else None
            )
        
        # Process citing_patent column
        if 'citing_patent' in df.columns:
            df['citing_patent'] = df['citing_patent'].apply(
                lambda x: x.split(', ') if pd.notna(x) else []
            )
            df['citing_patent'] = df['citing_patent'].apply(json.dumps)
        
        return df


class DatabaseManager:
    """Handles database operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.metadata = MetaData()
        self._create_engine()
        self._define_tables()
    
    def _create_engine(self):
        """Create database engine."""
        try:
            self.engine = create_engine(self.connection_string)
            logger.info("Database connection established")
        except SQLAlchemyError as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def _define_tables(self):
        """Define database table schemas."""
        self.patents_table = Table('patents', self.metadata,
            Column('lensid', String, primary_key=True),
            Column('patent_jurisdiction', String),
            Column('patent_publication_key', String),
            Column('patent_document_type', String),
            Column('patent_classification_codes', String),
            Column('patent_title', String),
            Column('patent_publication_date', Date),
            Column('patent_filing_date', Date),
            Column('patent_earliest_priority_date', Date),
            Column('patent_inventor', String),
            Column('patent_applicant', String),
            Column('patent_owner', String),
            Column('simple_patent_family_size', Integer),
            Column('backward_patent_citation_count', Integer),
            Column('forward_patent_citation_count', Integer)
        )
        
        self.citations_table = Table('patents_cit', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('doi', String),
            Column('citing_patent', JSON),
            Column('citation_string', String),
            Column('citation_external_id', String),
            Column('citing_family_count', Integer),
            Column('publisher', String),
            Column('journal', String),
            Column('publication_year', Integer),
            Column('publication_type', String),
            Column('subject_categories', String),
            Column('npl_author', String),
            Column('cited_by_scholarly_articles', Integer),
            Column('cited_by_patent_count', Integer),
            UniqueConstraint('doi', 'citation_string', name='uq_doi_citing')
        )
    
    def create_tables(self):
        """Create database tables."""
        try:
            self.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Insert DataFrame into database table."""
        try:
            df.to_sql(table_name, con=self.engine, index=False, if_exists=if_exists)
            logger.info(f"Data inserted into {table_name} table successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error inserting data into {table_name}: {e}")
            raise
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        try:
            return pd.read_sql_query(sql, self.engine)
        except SQLAlchemyError as e:
            logger.error(f"Error executing query: {e}")
            raise


class PatentAnalyzer:
    """Handles patent data analysis and visualization."""
    
    def __init__(self, db_manager: DatabaseManager, output_folder: str):
        self.db = db_manager
        self.output_folder = output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    def analyze_patents_by_instrument(self) -> pd.DataFrame:
        """Analyze patents grouped by instrument."""
        try:
            # Get patent citation data
            df_patent = self.db.query("SELECT doi, citing_patent FROM patents_cit")
            df_patent['citing_patent'] = df_patent['citing_patent'].apply(self._parse_json_array)
            
            # Group by DOI and flatten citing patents
            lensid_by_doi = (
                df_patent
                .groupby('doi')['citing_patent']
                .apply(lambda lists: list(set(chain.from_iterable(lists))))
                .reset_index()
            )
            print(lensid_by_doi)
            # Get flora data (assuming this table exists)
            try:
                df_flora = self.db.query("SELECT doi, title, year, instrument FROM flora_data")
                df_flora["instrument"] = df_flora["instrument"].str.split(r"[,/]")
                
                # Merge data
                df_merged = pd.merge(lensid_by_doi, df_flora[['doi', 'instrument']], on='doi', how='left')
                
                # Deduplicate by keeping row with most instruments
                df_merged['instrument_count'] = df_merged['instrument'].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
                df_sorted = df_merged.sort_values(by='instrument_count', ascending=False)
                df_dedup = df_sorted.drop_duplicates(subset='doi', keep='first')
                df_dedup = df_dedup.drop(columns='instrument_count')
                
                return df_dedup
                
            except Exception as e:
                logger.warning(f"Flora data not available: {e}")
                return lensid_by_doi
                
        except Exception as e:
            logger.error(f"Error in patent instrument analysis: {e}")
            raise
    
    def plot_patents_by_instrument(self, df: pd.DataFrame):
        """Create bar chart of patents by instrument."""
        try:
            # Create instrument-patent pairs
            instrument_patent_pairs = []
            for _, row in df.iterrows():
                if isinstance(row.get('instrument'), list) and isinstance(row.get('citing_patent'), list):
                    for inst in row['instrument']:
                        instrument_patent_pairs.append((inst, len(row['citing_patent'])))
            
            # Aggregate total patents per instrument
            inst_counter = Counter()
            for inst, count in instrument_patent_pairs:
                inst_counter[inst] += count
            
            # Convert to DataFrame for plotting
            inst_df = pd.DataFrame(inst_counter.items(), columns=['instrument', 'num_patents'])
            inst_df = inst_df.sort_values(by='num_patents', ascending=False)
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.bar(inst_df['instrument'], inst_df['num_patents'], color='skyblue')
            plt.xticks(rotation=90, ha='right')
            plt.xlabel('Instrument')
            plt.ylabel('Number of Patents (LensIDs)')
            plt.title('Number of Patents by Instrument')
            plt.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_folder, 'patents_by_instrument.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(f"Plot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating instrument plot: {e}")
            raise
    
    def plot_publication_patent_trends(self):
        """Plot publication and patent trends over time."""
        try:
            # Get publication data
            try:
                df_doi = self.db.query("SELECT * FROM process_doi")
                publication_counts = df_doi.groupby("year").agg(
                    in_flora_count=("in_flora", lambda x: (x == True).sum()),
                    missing_flora_count=("in_flora", lambda x: (x == False).sum())
                ).reset_index()
                publication_counts["total_publications"] = (
                    publication_counts["in_flora_count"] + publication_counts["missing_flora_count"]
                )
            except Exception:
                logger.warning("Publication data not available, skipping publication trend")
                publication_counts = pd.DataFrame()
            
            # Get patent data
            df_patent = self.db.query("SELECT lensid, patent_publication_date FROM patents")
            df_patent['year'] = pd.to_datetime(df_patent['patent_publication_date']).dt.year
            yearly_patent_trend = df_patent.groupby("year")["lensid"].count().reset_index()
            yearly_patent_trend.rename(columns={"lensid": "Patent Count"}, inplace=True)
            
            # Plot
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            # Plot publications if available
            if not publication_counts.empty:
                ax1.plot(publication_counts["year"], publication_counts["total_publications"], 
                        color="green", label="Total Publications", marker='o')
                ax1.plot(publication_counts["year"], publication_counts["in_flora_count"], 
                        color="red", label="Flora Publications", marker='+')
                ax1.set_ylabel("Publications", color="green")
                ax1.tick_params(axis="y", labelcolor="green")
                ax1.legend()
            
            # Plot patents
            ax2 = ax1.twinx()
            ax2.plot(yearly_patent_trend["year"], yearly_patent_trend["Patent Count"], 
                    color="blue", marker="o", label="Total Patents")
            ax2.set_ylabel("Patents", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")
            
            ax1.set_xlabel("Year")
            plt.title("Publications and Patents Over the Years")
            fig.tight_layout()
            
            # Save plot
            output_path = os.path.join(self.output_folder, 'publication_patent_trends.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(f"Plot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating trend plot: {e}")
            raise
    
    def plot_patent_document_types(self):
        """Create pie charts for patent document types."""
        try:
            df_patent = self.db.query("SELECT patent_document_type FROM patents")
            doc_counts = df_patent['patent_document_type'].value_counts()
            
            # Group small slices into "Other"
            top_n = 2
            major_types = doc_counts.nlargest(top_n)
            minor_types = doc_counts.drop(major_types.index)
            
            main_chart_data = major_types.copy()
            main_chart_data['Other'] = minor_types.sum()
            
            def autopct_label(values):
                def inner(pct):
                    total = sum(values)
                    val = int(round(pct * total / 100.0))
                    return f'{pct:.1f}%\n({val})'
                return inner
            
            # Main pie chart
            plt.figure(figsize=(8, 6))
            main_chart_data.plot.pie(
                autopct=autopct_label(main_chart_data),
                startangle=140,
                shadow=True,
                labeldistance=1.1,
                wedgeprops={'edgecolor': 'white'}
            )
            plt.title('Patent Document Type Distribution (Grouped)')
            plt.ylabel('')
            plt.tight_layout()
            
            output_path = os.path.join(self.output_folder, 'patent_types_main.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Breakdown of "Other"
            if len(minor_types) > 0:
                plt.figure(figsize=(7, 6))
                minor_types.plot.pie(
                    autopct=autopct_label(minor_types),
                    startangle=140,
                    shadow=True,
                    labeldistance=1.1,
                    wedgeprops={'edgecolor': 'white'}
                )
                plt.title('Breakdown of "Other" Patent Document Types')
                plt.ylabel('')
                plt.tight_layout()
                
                output_path = os.path.join(self.output_folder, 'patent_types_other.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.show()
            
            logger.info("Patent document type plots created successfully")
            
        except Exception as e:
            logger.error(f"Error creating patent type plots: {e}")
            raise
    
    @staticmethod
    def _parse_json_array(val):
        """Parse JSON array from database safely."""
        # Return empty list if value is NaN (and not a list)
        if isinstance(val, float) and pd.isna(val):
            return []
        
        # If it's already a list, just return it
        if isinstance(val, list):
            return val

        # If it's an ndarray with one string element, extract it
        if isinstance(val, np.ndarray) and val.size == 1:
            val = val.item()

        # Try to load the JSON string
        try:
            return json.loads(val) if isinstance(val, str) else val
        except (json.JSONDecodeError, TypeError):
            return []

class PatentPipeline:
    """Main pipeline class that orchestrates the entire process."""
    
    def __init__(self, config: Config):
        self.config = config
        self.scraper = WebScraper(config)
        self.processor = DataProcessor()
        self.db_manager = DatabaseManager(config.db_connection_string)
        self.analyzer = PatentAnalyzer(self.db_manager, config.output_folder)
    
    def run_full_pipeline(self):
        """Run the complete patent data pipeline."""
        try:
            logger.info("Starting patent data pipeline...")
            
            # Step 1: Scrape data
            # logger.info("Step 1: Scraping patent data from Lens.org...")
            # patent_file, citation_file = self.scraper.scrape_patent_data(self.config.input_file_path)
            excel_files = sorted([f for f in os.listdir(self.config.download_folder) if f.endswith('.xlsx')])
            citation_file, patent_file =[os.path.join(self.config.download_folder, f) for f in excel_files[:2]]

            # Step 2: Process data
            logger.info("Step 2: Processing downloaded data...")
            df_patents = pd.read_excel(patent_file)
            df_citations = pd.read_excel(citation_file)
            
            df_patents = self.processor.preprocess_dataframe(df_patents)
            df_citations = self.processor.preprocess_dataframe(df_citations)
            df_citations = self.processor.process_citation_data(df_citations)
            print(df_patents.columns, df_citations.columns)
            # Step 3: Store in database
            logger.info("Step 3: Storing data in database...")
            self.db_manager.create_tables()
            self.db_manager.insert_dataframe(df_patents, 'patents')
            self.db_manager.insert_dataframe(df_citations, 'patents_cit')
            
            # Step 4: Analyze and visualize
            logger.info("Step 4: Analyzing data and creating visualizations...")
            
            # Analyze patents by instrument
            df_instrument_analysis = self.analyzer.analyze_patents_by_instrument()
            self.analyzer.plot_patents_by_instrument(df_instrument_analysis)
            
            # Plot trends
            self.analyzer.plot_publication_patent_trends()
            
            # Plot patent types
            self.analyzer.plot_patent_document_types()
            
            logger.info("Patent data pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def run_analysis_only(self):
        """Run only the analysis part (assumes data is already in database)."""
        try:
            logger.info("Running analysis on existing data...")
            
            df_instrument_analysis = self.analyzer.analyze_patents_by_instrument()
            self.analyzer.plot_patents_by_instrument(df_instrument_analysis)
            self.analyzer.plot_publication_patent_trends()
            self.analyzer.plot_patent_document_types()
            
            logger.info("Analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main function to run the patent pipeline."""
    # Example configuration
    config_dict = {
        "URL": "https://www.lens.org/lens/patcite",
        "download_folder": r"C:\Users\vu-hong\Desktop\davies\test_data",
        "input_file_path": r"C:\Users\vu-hong\Desktop\ILL_corpus\Citation_Network\list_doi.txt",
        "db_connection_string": "postgresql://postgres:1234@localhost:5432/ill_corpus",
        "output_folder": r"C:\Users\vu-hong\Desktop\davies\test_data"
    }
    
    config = Config(config_dict)
    pipeline = PatentPipeline(config)
    
    # Run the full pipeline
    try:
        pipeline.run_full_pipeline()
        # pipeline.run_analysis_only()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        # Optionally, try running analysis only if data exists
        logger.info("Attempting to run analysis on existing data...")
        try:
            pipeline.run_analysis_only()
        except Exception as analysis_error:
            logger.error(f"Analysis also failed: {analysis_error}")


if __name__ == "__main__":
    main()