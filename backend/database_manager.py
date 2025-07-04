import os
import json
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, Column, String, Integer, Boolean, DateTime, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
import re
import aiohttp
import asyncio
from abc import ABC, abstractmethod
from sqlalchemy.orm import selectinload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()

class DatabaseConfig:
    """Database configuration class using config.json"""
    def __init__(self, config_file: str = "config.json"):
        self.config = self._load_config(config_file)
        self.database_config = self.config.get('Database', {})
        self.database_uri = self._get_database_uri()
        self.echo = self.database_config.get('echo', False)

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.error(f"Config file '{config_file}' not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in '{config_file}'")
            raise

    def _get_database_uri(self) -> str:
        """Get or construct database URI from config"""
        if 'URI' in self.database_config:
            return self.database_config['URI']
        # Construct URI from individual components
        dbname = self.database_config.get('dbname', 'test0')
        user = self.database_config.get('user', 'postgres')
        password = self.database_config.get('password', '1234')
        host = self.database_config.get('host', 'localhost')
        port = self.database_config.get('port', '5432')
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

class FloraPublication(Base):
    """Model for Flora publication data"""
    __tablename__ = 'flora_data'
    
    id = Column(String, primary_key=True)
    doi = Column(String, unique=True, nullable=True)
    title = Column(String, nullable=True)
    source = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    instrument = Column(String, nullable=True)
    last_update = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<FloraPublication(id='{self.id}', doi='{self.doi}', title='{self.title[:50]}...')>"

class OpenAlexPublication(Base):
    """Model for OpenAlex publication data"""
    __tablename__ = 'openalex_data'
    
    id = Column(String, primary_key=True)
    doi = Column(String, unique=True, nullable=True)
    title = Column(String, nullable=True)
    type = Column(String, nullable=True)
    source = Column(String, nullable=True)
    cite_count = Column(Integer, nullable=True)
    year = Column(Integer, nullable=True)
    last_update = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<OpenAlexPublication(id='{self.id}', doi='{self.doi}', citations={self.cite_count})>"

class ProcessedDOI(Base):
    """Model for processed DOI data"""
    __tablename__ = 'process_doi'
    
    doi = Column(String, primary_key=True)
    in_flora = Column(Boolean, default=False)
    in_openalex = Column(Boolean, default=False)
    source_count = Column(Integer, default=0)
    tier = Column(String, nullable=True)
    instrument = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    last_update = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<ProcessedDOI(doi='{self.doi}', tier='{self.tier}', sources={self.source_count})>"

class D0Works(Base):
    """Model for analysis data"""
    __tablename__ = 'd0_works'
    
    doi = Column(String, primary_key=True)
    tier = Column(String, nullable=True)
    instrument = Column(String, nullable=True)
    data = Column(JSON, nullable=True)
    last_update = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<D0Works(doi='{self.doi}', tier='{self.tier}')>"

class D0Citation(Base):
    """Model for citation data"""
    __tablename__ = 'd0_citation'

    id = Column(String, primary_key=True)
    doi = Column(String)
    tier = Column(String)
    publication_year = Column(Integer)
    cited_by_count = Column(Integer)
    cited_by_api_url = Column(String)
    counts_by_year = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<D0Citation(id='{self.id}', doi='{self.doi}', year={self.publication_year})>"

class D1Citation(Base):
    """Model for citation data"""
    __tablename__ = 'd1_citation'

    id = Column(String, primary_key=True)
    doi = Column(String)
    cites_d0_id = Column(String)
    tier = Column(String)
    publication_year = Column(Integer)
    cited_by_count = Column(Integer)
    cited_by_api_url = Column(String)
    counts_by_year = Column(JSON)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<D1Citation(id='{self.id}', doi='{self.doi}', year={self.publication_year})>"

class DOICleaner:
    """Utility class for cleaning and normalizing DOIs"""
    
    @staticmethod
    def clean_doi(doi: Optional[str]) -> Optional[str]:
        """Clean and normalize DOI string"""
        if doi is None or not isinstance(doi, str):
            return None
            
        doi = doi.strip().lower()
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)
        prefixes = ['https://doi.org/', 'http://doi.org/', 'doi.org/', 'doi:']
        for prefix in prefixes:
            if doi.startswith(prefix):
                doi = doi.replace(prefix, '', 1)
                break
        return doi if doi else None

class DatabaseManager:
    """Main database manager class for handling all database operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = DatabaseConfig(config_file=config) if isinstance(config, str) else config or DatabaseConfig()
        self.engine = None
        self.Session = None
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database engine and session factory"""
        try:
            self.engine = create_engine(
                self.config.database_uri,
                echo=self.config.echo,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("All tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def insert_flora_data(self, flora_records: List[Dict[str, Any]]) -> int:
        """Insert Flora publication data"""
        try:
            with self.get_session() as session:
                inserted_count = 0
                for record in flora_records:
                    flora_pub = FloraPublication(
                    # stmt = insert(FloraPublication).values(
                        id=record.get('id'),
                        doi=DOICleaner.clean_doi(record.get('doi')),
                        title=record.get('title'),
                        source=record.get('source'),
                        year=record.get('year'),
                        instrument=record.get('instrument')
                    )
                    session.merge(flora_pub)
                    # self.session.execute(stmt)
                    inserted_count += 1
                logger.info(f"Inserted/updated {inserted_count} Flora records")
                return inserted_count
        except Exception as e:
            logger.error(f"Error inserting Flora data: {e}")
            raise

    def insert_openalex_data(self, openalex_records: List[Dict[str, Any]]) -> int:
        """Insert OpenAlex publication data"""
        try:
            with self.get_session() as session:
                inserted_count = 0
                for record in openalex_records:
                    openalex_pub = OpenAlexPublication(
                    # stmt = insert(OpenAlexPublication).values(
                        id=record.get('id'),
                        doi=DOICleaner.clean_doi(record.get('doi')),
                        title=record.get('title'),
                        type=record.get('type'),
                        source=record.get('source'),
                        cite_count=record.get('cite_count'),
                        year=record.get('year')
                    )
                    session.merge(openalex_pub)
                    # self.session.execute(stmt)
                    inserted_count += 1
                logger.info(f"Inserted/updated {inserted_count} OpenAlex records")
                return inserted_count
        except Exception as e:
            logger.error(f"Error inserting OpenAlex data: {e}")
            raise

    def store_data_by_source(self, source: str, data: List[Dict[str, Any]]) -> int:
        """Store data based on the source"""
        store_methods = {
            'Flora': self.insert_flora_data,
            'OpenAlex': self.insert_openalex_data
        }
        if source not in store_methods:
            raise ValueError(f"Unknown source: {source}")
        return store_methods[source](data)

    def get_flora_data(self, limit: int = None) -> List[FloraPublication]:
        """Get Flora publication data"""
        try:
            with self.get_session() as session:
                query = session.query(FloraPublication)
                if limit:
                    query = query.limit(limit)
                return query.all()
        except Exception as e:
            logger.error(f"Error querying Flora data: {e}")
            raise

    def get_openalex_data(self, limit: int = None) -> List[OpenAlexPublication]:
        """Get OpenAlex publication data"""
        try:
            with self.get_session() as session:
                query = session.query(OpenAlexPublication)
                if limit:
                    query = query.limit(limit)
                return query.all()
        except Exception as e:
            logger.error(f"Error querying OpenAlex data: {e}")
            raise

    # UTILITY METHODS
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with self.get_session() as session:
                stats = {
                    "flora_count": session.query(FloraPublication).count(),
                    "openalex_count": session.query(OpenAlexPublication).count(),
                    "processed_dois_count": session.query(ProcessedDOI).count()
                }
                return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise

class CitationNetworkDatabase(DatabaseManager):
    """Extended database manager for citation network operations"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._create_citation_tables()
    
    def _create_citation_tables(self):
        """Create citation network specific tables"""
        try:
            D0Works.metadata.create_all(self.engine)
            D0Citation.metadata.create_all(self.engine)
            D1Citation.metadata.create_all(self.engine)
            logger.info("Citation network tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create citation tables: {e}")
            raise
    
    def insert_d0_works(self, works_data: List[Dict[str, Any]]) -> int:
        """Insert D0 works data"""
        try:
            with self.get_session() as session:
                inserted_count = 0
                for work in works_data:
                    d0_work = D0Works(
                        doi=DOICleaner.clean_doi(work.get('doi')),
                        tier=work.get('tier'),
                        data=work.get('data')
                    )
                    session.merge(d0_work)
                    inserted_count += 1
                logger.info(f"Inserted/updated {inserted_count} D0 works")
                return inserted_count
        except Exception as e:
            logger.error(f"Error inserting D0 works: {e}")
            raise
    
    def insert_d0_citations(self, citations_data: List[Dict[str, Any]]) -> int:
        """Insert D0 citation data"""
        try:
            with self.get_session() as session:
                inserted_count = 0
                for citation in citations_data:
                    d0_citation = D0Citation(
                        id=citation.get('id'),
                        doi=DOICleaner.clean_doi(citation.get('doi')),
                        publication_year=citation.get('publication_year'),
                        cited_by_count=citation.get('cited_by_count'),
                        cited_by_api_url=citation.get('cited_by_api_url'),
                        counts_by_year=citation.get('counts_by_year'),
                        tier=citation.get('tier')
                    )
                    session.merge(d0_citation)
                    inserted_count += 1
                logger.info(f"Inserted/updated {inserted_count} D0 citations")
                return inserted_count
        except Exception as e:
            logger.error(f"Error inserting D0 citations: {e}")
            raise
    
    def insert_d1_citations(self, citations_data: List[Dict[str, Any]]) -> int:
        """Insert D1 citation data"""
        try:
            with self.get_session() as session:
                inserted_count = 0
                for citation in citations_data:
                    d1_citation = D1Citation(
                        id=citation.get('id'),
                        doi=DOICleaner.clean_doi(citation.get('doi')),
                        cites_d0_id=citation.get('cites_d0_id'),
                        tier=citation.get('tier'),
                        publication_year=citation.get('publication_year'),
                        cited_by_count=citation.get('cited_by_count'),
                        cited_by_api_url=citation.get('cited_by_api_url'),
                        counts_by_year=citation.get('counts_by_year'),
                        data=citation.get('data')
                    )
                    session.merge(d1_citation)
                    inserted_count += 1
                logger.info(f"Inserted/updated {inserted_count} D1 citations")
                return inserted_count
        except Exception as e:
            logger.error(f"Error inserting D1 citations: {e}")
            raise
    
    def get_d0_citations_for_processing(self, work_filter: str = "cited_by_count > 0") -> List[Dict]:
        """Get D0 citations for further processing"""
        try:
            with self.get_session() as session:
                query = f"SELECT * FROM d0_citation WHERE {work_filter}"
                result = session.execute(text(query))
                return list(result.mappings())
        except Exception as e:
            logger.error(f"Error getting D0 citations: {e}")
            raise

    def get_processed_dois(self, tier: str = None) -> List[ProcessedDOI]:
        """Get processed DOI data"""
        try:
            with self.get_session() as session:
                query = session.query(ProcessedDOI).options(selectinload('*'))  # or specify relationships explicitly
                if tier:
                    query = query.filter(ProcessedDOI.tier == tier)
                results = query.all()
                # At this point, all needed data is loaded
                # return results
                return [
                {
                    'doi': p.doi,
                    'tier': p.tier,
                    'instrument': p.instrument,
                    # Add any other fields you need
                }
                for p in results
            ]
        except Exception as e:
            logger.error(f"Error querying processed DOIs: {e}")
            raise

class DataProcessor:
    """Class for processing data, such as DOI reconciliation"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def process_dois(self) -> int:
        """Process DOIs by comparing Flora and OpenAlex data"""
        try:
            with self.db_manager.get_session() as session:
                flora_pubs = session.query(FloraPublication).filter(
                    FloraPublication.doi.isnot(None),
                    FloraPublication.doi != ''
                ).all()
                
                openalex_pubs = session.query(OpenAlexPublication).filter(
                    OpenAlexPublication.doi.isnot(None),
                    OpenAlexPublication.doi != ''
                ).all()
                
                flora_mappings = {pub.doi: pub for pub in flora_pubs}
                openalex_mappings = {pub.doi: pub for pub in openalex_pubs}
                
                all_dois = set(flora_mappings.keys()) | set(openalex_mappings.keys())
                
                year_mappings = {}
                for doi in all_dois:
                    flora_pub = flora_mappings.get(doi)
                    openalex_pub = openalex_mappings.get(doi)
                    if flora_pub and flora_pub.year:
                        year_mappings[doi] = flora_pub.year
                    elif openalex_pub and openalex_pub.year:
                        year_mappings[doi] = openalex_pub.year
                    if (flora_pub and flora_pub.year and 
                        openalex_pub and openalex_pub.year and 
                        flora_pub.year != openalex_pub.year):
                        logger.warning(f"Year conflict for DOI {doi}: Flora={flora_pub.year}, OpenAlex={openalex_pub.year}")
                
                processed_count = 0
                for doi in all_dois:
                    if not doi or not doi.strip():
                        continue
                    
                    flora_info = flora_mappings.get(doi)
                    openalex_info = openalex_mappings.get(doi)
                    
                    in_flora = doi in flora_mappings
                    in_openalex = doi in openalex_mappings
                    source_count = sum([in_flora, in_openalex])
                    
                    year = year_mappings.get(doi)
                    tier = self._determine_tier_from_flora(flora_info, openalex_info)
                    instrument = flora_info.instrument if flora_info and flora_info.instrument else None
                    
                    existing_processed = session.query(ProcessedDOI).filter_by(doi=doi).first()
                    
                    if existing_processed:
                        existing_processed.in_flora = in_flora
                        existing_processed.in_openalex = in_openalex
                        existing_processed.source_count = source_count
                        existing_processed.tier = tier
                        existing_processed.instrument = instrument
                        existing_processed.year = year
                        existing_processed.last_update = datetime.utcnow()
                    else:
                        processed_doi = ProcessedDOI(
                            doi=doi,
                            in_flora=in_flora,
                            in_openalex=in_openalex,
                            source_count=source_count,
                            tier=tier,
                            instrument=instrument,
                            year=year
                        )
                        session.add(processed_doi)
                    
                    processed_count += 1
                
                logger.info(f"Processed {processed_count} DOIs")
                return processed_count
        except Exception as e:
            logger.error(f"Error processing DOIs: {e}")
            raise
    
    def get_processed_dois(self, tier: str = None) -> List[ProcessedDOI]:
        """Get processed DOI data"""
        try:
            with self.get_session() as session:
                query = session.query(ProcessedDOI)
                if tier:
                    query = query.filter(ProcessedDOI.tier == tier)
                return query.all()
        except Exception as e:
            logger.error(f"Error querying processed DOIs: {e}")
            raise

    def _determine_tier_from_flora(self, flora_info, openalex_info) -> str:
        """Determine tier based on Flora document type"""
        if flora_info:
            doc_type = self._label_doc_type(flora_info.id)
            if doc_type == "tier1":
                return "tier1"
            return "tier2"
        return "tier2"

    def _label_doc_type(self, flora_id: str) -> str:
        """Determine document type based on Flora ID pattern"""
        if flora_id.endswith("T"):
            return "Technical Reports"
        elif flora_id.startswith("ILL") and len(flora_id) >= 8:
            if flora_id[7] == "4" and len(flora_id) == 11:
                return "Book"
            elif flora_id[7] == "2" and len(flora_id) == 11:
                return "Theses"
            elif flora_id[7] == "7" and len(flora_id) == 11:
                return "Articles citing the ILL"
            elif flora_id[7] == "3" and len(flora_id) == 11:
                return "Workshop"
            return "tier1"
        return "Unknown"
    

# Example usage and testing
if __name__ == "__main__":

    CONFIG_FILE = r"C:\Users\vu-hong\Desktop\davies\web_biblio\backend\config.json"
    # Initialize database manager
    db_manager = DatabaseManager(CONFIG_FILE)
    
    # Create tables
    db_manager.create_tables()

    # Example data insertion
    sample_flora_data = [
        {
            "id": "flora_001",
            "doi": "10.1000/sample1",
            "title": "Sample Flora Publication 1",
            "source": "Flora Journal",
            "year": 2023,
            "instrument": "Telescope A"
        }
    ]
    
    sample_openalex_data = [
        {
            "id": "openalex_001",
            "doi": "10.1000/sample1",
            "title": "Sample OpenAlex Publication 1",
            "type": "journal-article",
            "source": "Nature",
            "cite_count": 150,
            "year": 2023
        }
    ]

    try:
        # Step 1: Insert data
        # db_manager.insert_flora_data(sample_flora_data)
        # db_manager.insert_openalex_data(sample_openalex_data)
        db_manager.store_data_by_source("Flora", sample_flora_data)
        db_manager.store_data_by_source("OpenAlex", sample_openalex_data)

        # Step 2: Process data
        db_manager.process_dois()

        # Step 4: Get statistics
        stats = db_manager.get_statistics()
        print("Database Statistics:", stats)
    except Exception as e:
        logger.error(f"Error in workflow: {e}")