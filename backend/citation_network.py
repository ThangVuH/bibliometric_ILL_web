import asyncio
import aiohttp
import json
import pandas as pd
import re
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from contextlib import asynccontextmanager
import random
from tqdm.asyncio import tqdm_asyncio

from database_manager import D0Works, D0Citation, D1Citation, DOICleaner, CitationNetworkDatabase
from database_manager import DatabaseManager, DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAlexAPIClient:
    """Base class for OpenAlex API interactions with rate limiting and retry logic"""
    
    def __init__(self, email: str, api_key: Optional[str] = None, 
                 max_connections: int = 10, max_retries: int = 3):
        self.email = email
        self.api_key = api_key
        self.headers = {
            "User-Agent": f"CitationNetworkBuilder/1.0 ({email})"
        }
        self.semaphore = asyncio.Semaphore(max_connections)
        self.max_retries = max_retries
        
    async def make_request(self, session: aiohttp.ClientSession, url: str, 
                          params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request with retry logic and rate limiting"""
        if params is None:
            params = {}
        if self.api_key:
            params['api_key'] = self.api_key
            
        for attempt in range(self.max_retries):
            try:
                async with self.semaphore:
                    async with session.get(url, headers=self.headers, params=params) as response:
                        if response.status == 429:  # Rate limit
                            wait_time = random.uniform(2, 5) * (2 ** attempt)
                            logger.warning(f"Rate limited. Waiting {wait_time:.2f}s")
                            await asyncio.sleep(wait_time)
                            continue
                            
                        response.raise_for_status()
                        return await response.json()
                        
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = random.uniform(3, 5) * (2 ** attempt)
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Final failure for URL {url}: {e}")
                    
        return None


class D0WorksFetcher:
    """Fetches publication metadata from OpenAlex (D0 works)"""
    
    def __init__(self, db: CitationNetworkDatabase, api_client: OpenAlexAPIClient):
        self.db = db
        self.api_client = api_client
        self.api_base_url = "https://api.openalex.org/works"
        
    async def fetch_single_work(self, session: aiohttp.ClientSession, 
                               doi: str, tier: str = None) -> Optional[Dict]:
        """Fetch a single work by DOI"""
        clean_doi = DOICleaner.clean_doi(doi)
        if not clean_doi:
            logger.warning(f"Invalid DOI: {doi}")
            return None
            
        url = f"{self.api_base_url}/doi:{clean_doi}"
        data = await self.api_client.make_request(session, url)
        
        if data and 'error' not in data:
            return {
                'doi': clean_doi,
                'tier': tier,
                'data': data
            }
        else:
            logger.warning(f"No valid data for DOI: {clean_doi}")
            return None
    
    async def fetch_all_works(self, doi_list: List[str], tier_list: List[str] = None, 
                             batch_size: int = 100) -> Tuple[int, int]:
        """Fetch all works from DOI list"""
        if tier_list is None:
            tier_list = [None] * len(doi_list)
        elif len(tier_list) != len(doi_list):
            raise ValueError("DOI list and tier list must have the same length")
            
        logger.info(f"Starting to fetch {len(doi_list)} works")
        
        success_count = 0
        failure_count = 0
        
        async with aiohttp.ClientSession() as session:
            # Process in batches
            for i in tqdm_asyncio(range(0, len(doi_list), batch_size), desc="Fetching D0 Works"):
                doi_batch = doi_list[i:i + batch_size]
                tier_batch = tier_list[i:i + batch_size]
                
                # Create tasks for the current batch
                tasks = [
                    self.fetch_single_work(session, doi, tier)
                    for doi, tier in zip(doi_batch, tier_batch)
                ]
                
                # Run tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results and store in database
                works_data = []
                for doi, result in zip(doi_batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing DOI {doi}: {result}")
                        failure_count += 1
                    elif result is not None:
                        works_data.append(result)
                        success_count += 1
                    else:
                        failure_count += 1
                
                # Store batch in database
                if works_data:
                    self.db.insert_d0_works(works_data)
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.1)
        
        logger.info(f"Finished fetching works. Success: {success_count}, Failures: {failure_count}")
        return success_count, failure_count
    
    async def fetch_works_from_processed_dois(self, tier: str = None, 
                            batch_size: int = 100) -> Tuple[int, int]:
        """Fetch D0 works from processed DOIs table"""
        logger.info(f"Fetching processed DOIs from database{' for tier: ' + tier if tier else ''}")
        
        try:
            # Get processed DOIs from database
            processed_dois = self.db.get_processed_dois(tier=tier)
            
            if not processed_dois:
                logger.warning(f"No processed DOIs found{' for tier: ' + tier if tier else ''}")
                return 0, 0
            
            logger.info(f"Found {len(processed_dois)} processed DOIs to fetch")

            # Extract DOI and tier information
            doi_list = []
            tier_list = []
            
            # for processed_doi in processed_dois:
            #     if processed_doi.doi:  # Make sure DOI exists
            #         doi_list.append(processed_doi.doi)
            #         tier_list.append(processed_doi.tier)
            
            # if not doi_list:
            #     logger.warning("No valid DOIs found in processed DOIs")
            #     return 0, 0
            
            for processed_doi in processed_dois:
                doi, tier_value = processed_doi['doi'], processed_doi['tier']
                # print(doi, tier_value)
                if doi:  # Make sure DOI exists
                    doi_list.append(doi)
                    tier_list.append(tier_value)
            
            if not doi_list:
                logger.warning("No valid DOIs found in processed DOIs")
                return 0, 0
            
            # Use the existing fetch_all_works method
            return await self.fetch_all_works(doi_list, tier_list, batch_size)
        except Exception as e:
            logger.error(f"Error fetching works from processed DOIs: {e}")
            raise

    async def fetch_works_from_csv(self, csv_file_path: str, 
                                  doi_column: str = 'doi', 
                                  tier_column: str = None, 
                                  batch_size: int = 100) -> Tuple[int, int]:
        """Fetch D0 works from CSV file (kept for backward compatibility)"""
        logger.info(f"Loading DOIs from {csv_file_path}")
        
        df = pd.read_csv(csv_file_path)
        doi_list = df[doi_column].dropna().tolist()
        tier_list = df[tier_column].tolist() if tier_column and tier_column in df.columns else None
        
        return await self.fetch_all_works(doi_list, tier_list, batch_size)


class D0CitationProcessor:
    """Processes D0 works data into citation format"""
    
    def __init__(self, db: CitationNetworkDatabase):
        self.db = db
        
    def process_d0_citations(self) -> int:
        """Extract citation data from D0 works and save to D0 citation table"""
        logger.info("Processing D0 works into citation format")
        
        try:
            with self.db.get_session() as session:
                # Get all D0 works
                d0_works = session.query(D0Works).all()
                
                if not d0_works:
                    logger.warning("No D0 works data found")
                    return 0
                
                citations_data = []
                for work in d0_works:
                    if work.data and 'error' not in work.data:
                        citation_data = {
                            'id': work.data.get('id', '').split('/')[-1] if work.data.get('id') else None,
                            'doi': DOICleaner.clean_doi(work.data.get('doi')),
                            'publication_year': work.data.get('publication_year'),
                            'cited_by_count': work.data.get('cited_by_count'),
                            'cited_by_api_url': work.data.get('cited_by_api_url'),
                            'counts_by_year': work.data.get('counts_by_year'),
                            'tier': work.tier
                        }
                        citations_data.append(citation_data)
                
                # Insert processed citations
                processed_count = self.db.insert_d0_citations(citations_data)
                logger.info(f"Processed {processed_count} D0 citations")
                return processed_count
                
        except Exception as e:
            logger.error(f"Error processing D0 citations: {e}")
            raise


class D1CitationFetcher:
    """Fetches citations of D0 works (D1 citations)"""
    
    def __init__(self, db: CitationNetworkDatabase, api_client: OpenAlexAPIClient):
        self.db = db
        self.api_client = api_client
        
    async def fetch_citations_for_work(self, session: aiohttp.ClientSession, 
                                     work_data: Dict) -> Tuple[int, int]:
        """Fetch all citations for a single D0 work"""
        cursor = "*"
        base_url = work_data.get("cited_by_api_url")
        d0_id = work_data.get("id")
        d0_tier = work_data.get("tier")
        
        if not base_url or not d0_id:
            logger.warning(f"Missing required data for work {d0_id}")
            return 0, 1
            
        logger.debug(f"Fetching citations for work {d0_id}")
        citation_count = 0
        citations_batch = []
        
        while cursor:
            params = {"cursor": cursor, "per_page": 200}
            data = await self.api_client.make_request(session, base_url, params)
            
            if not data:
                break
                
            # Process citations in this page
            for item in data.get("results", []):
                citation_data = {
                    'id': item.get('id', '').split('/')[-1] if item.get('id') else None,
                    'doi': DOICleaner.clean_doi(item.get('doi')),
                    'cites_d0_id': d0_id,
                    'tier': d0_tier,
                    'publication_year': item.get('publication_year'),
                    'cited_by_count': item.get('cited_by_count'),
                    'cited_by_api_url': item.get('cited_by_api_url'),
                    'counts_by_year': item.get('counts_by_year'),
                    'data': item
                }
                citations_batch.append(citation_data)
                citation_count += 1
            
            cursor = data.get("meta", {}).get("next_cursor")
        
        # Store citations in database
        if citations_batch:
            self.db.insert_d1_citations(citations_batch)
        
        logger.debug(f"Fetched {citation_count} citations for work {d0_id}")
        return citation_count, 0  # success_count, failure_count
    
    async def fetch_all_citations(self, work_filter: str = "cited_by_count > 0", 
                                 batch_size: int = 100) -> Tuple[int, int, int, int]:
        """Fetch citations for all D0 works matching the filter"""
        
        # Get D0 citation data for processing
        works_data = self.db.get_d0_citations_for_processing(work_filter)
        
        if not works_data:
            logger.warning("No D0 citation data found matching filter")
            return 0, 0, 0, 0
            
        logger.info(f"Fetching D1 citations for {len(works_data)} works")
        
        total_cited_by_count = sum(work.get('cited_by_count', 0) for work in works_data)
        logger.info(f"Total expected citations: {total_cited_by_count}")
        
        success_count = 0
        failure_count = 0
        total_citation_count = 0
        
        async with aiohttp.ClientSession() as session:
            # Process works in batches
            for i in tqdm_asyncio(range(0, len(works_data), batch_size), desc="Fetching D1 Citations"):
                works_batch = works_data[i:i + batch_size]
                
                # Create tasks for the current batch
                tasks = [
                    self.fetch_citations_for_work(session, work_data)
                    for work_data in works_batch
                ]
                
                # Run tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for work_data, result in zip(works_batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing work {work_data.get('id', 'unknown')}: {result}")
                        failure_count += 1
                    else:
                        citations_fetched, _ = result
                        total_citation_count += citations_fetched
                        success_count += 1
                
                # Small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)
        
        logger.info(f"Finished fetching D1 citations. Success: {success_count}, Failures: {failure_count}")
        return success_count, failure_count, total_cited_by_count, total_citation_count


class CitationNetworkBuilder:
    """Main orchestrator for building citation networks"""
    
    # def __init__(self, db_config: Optional[DatabaseConfig] = None, 
    def __init__(self, db_config = None, 
                 email: str = None, api_key: str = None):
        if not email:
            raise ValueError("Email is required for OpenAlex API")
            
        self.db = CitationNetworkDatabase(db_config)
        self.api_client = OpenAlexAPIClient(email, api_key)
        self.d0_fetcher = D0WorksFetcher(self.db, self.api_client)
        self.d0_processor = D0CitationProcessor(self.db)
        self.d1_fetcher = D1CitationFetcher(self.db, self.api_client)
        
    async def build_d0_works_from_processed_dois(self, tier: str = None, 
                                               batch_size: int = 100) -> Tuple[int, int]:
        """Build D0 works from processed DOIs table"""
        return await self.d0_fetcher.fetch_works_from_processed_dois(tier, batch_size)
    
    async def build_d0_works_from_csv(self, csv_file_path: str, 
                                     doi_column: str = 'doi', 
                                     tier_column: str = None, 
                                     batch_size: int = 100) -> Tuple[int, int]:
        """Build D0 works from CSV file containing DOIs"""
        logger.info(f"Loading DOIs from {csv_file_path}")
        
        df = pd.read_csv(csv_file_path)
        doi_list = df[doi_column].dropna().tolist()
        tier_list = df[tier_column].tolist() if tier_column and tier_column in df.columns else None
        
        return await self.d0_fetcher.fetch_all_works(doi_list, tier_list, batch_size)
        
    def process_d0_citations(self) -> int:
        """Process D0 works into citation format"""
        return self.d0_processor.process_d0_citations()
        
    async def build_d1_citations(self, work_filter: str = "cited_by_count > 0", 
                                batch_size: int = 100) -> Tuple[int, int, int, int]:
        """Build D1 citations for works matching the filter"""
        return await self.d1_fetcher.fetch_all_citations(work_filter, batch_size)
        
    async def build_full_network(self, csv_file_path: str, 
                                doi_column: str = 'doi', 
                                tier_column: str = None, 
                                work_filter: str = "cited_by_count > 0") -> Dict[str, Any]:
        """Build the complete citation network and return statistics"""
        logger.info("Starting full citation network build")
        
        # Step 1: Fetch D0 works
        d0_success, d0_failure = await self.build_d0_works_from_csv(
            csv_file_path, doi_column, tier_column
        )
        
        # Step 2: Process D0 citations
        d0_processed = self.process_d0_citations()
        
        # Step 3: Fetch D1 citations
        d1_success, d1_failure, expected_citations, actual_citations = await self.build_d1_citations(work_filter)
        
        # Compile statistics
        stats = {
            'd0_works': {
                'success': d0_success,
                'failure': d0_failure,
                'total': d0_success + d0_failure
            },
            'd0_citations_processed': d0_processed,
            'd1_citations': {
                'works_success': d1_success,
                'works_failure': d1_failure,
                'expected_citations': expected_citations,
                'actual_citations': actual_citations
            }
        }
        
        logger.info("Citation network build completed")
        logger.info(f"Build statistics: {stats}")
        
        return stats
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        base_stats = self.db.get_statistics()
        
        # Add citation-specific statistics
        try:
            with self.db.get_session() as session:
                d0_works_count = session.query(D0Works).count()
                d0_citations_count = session.query(D0Citation).count()
                d1_citations_count = session.query(D1Citation).count()
                
                citation_stats = {
                    'd0_works_count': d0_works_count,
                    'd0_citations_count': d0_citations_count,
                    'd1_citations_count': d1_citations_count
                }
                
                return {**base_stats, **citation_stats}
        except Exception as e:
            logger.error(f"Error getting network statistics: {e}")
            return base_stats
