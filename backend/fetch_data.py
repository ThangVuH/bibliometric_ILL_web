import json
import os
import copy
import logging
from schema import SchemaError

from lib_api import OpenAlexAPI, FloraAPI, ScopusAPI
# from lib_scrap import WorkflowManager

def load_config(config_path="config.json"):
    """Load and validate configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in '{config_path}'.")
    except SchemaError as e:
        raise ValueError(f"Config validation failed: {e}")
    

def generate_file_path(api_name, config, year):
    """Generate file path for CSV output, including the year in the file name."""
    download_folder = config[api_name]["download_folder"]
    file_name = f"{api_name}_{year}.csv"  # Include year in file name
    return os.path.join(download_folder, file_name)

def extract_data_for_years(config, start_year, end_year, db_ops=None):
    """
    Extract data for a range of years from multiple sources and store in database.
    
    Args:
        config: Configuration dictionary
        start_year: Start year for data extraction
        end_year: End year for data extraction
        db_ops: DatabaseOperations instance for storing data
    """
    for year in range(start_year, end_year + 1):
        logging.info(f"Processing year {year}...")

        # Process OpenAlex, Flora, Scopus
        for api_name in ['Flora', 'OpenAlex']:
        # for api_name in ['OpenAlex', 'Flora', 'Scopus']:
            # Create a deep copy of the config for this API and year
            temp_config = copy.deepcopy(config)
            temp_config[api_name]['PUBLICATION_YEAR'] = year
            
            try:
                if api_name == 'OpenAlex':
                    queries = [
                        {"filter": f"institutions.ror:{temp_config['OpenAlex']['ROR']},publication_year:{year}", "per-page": temp_config['OpenAlex']["PER_PAGE"]},
                        {"filter": f"authorships.institutions.lineage:!{temp_config['OpenAlex']['INSTITUTION_ID']},publication_year:{year},default.search:{temp_config['OpenAlex']['query']}", "per-page": temp_config['OpenAlex']["PER_PAGE"]}
                    ]
                    api = OpenAlexAPI(temp_config['OpenAlex'])
                    data = api.fetch_data(queries)
                    normalized_data = api.normalize_data(data)
                
                elif api_name == 'Flora':
                    # Update the Flora query for the current year
                    query_template = temp_config['Flora']['parameters_query']['query']
                    query = query_template.replace("{year}", str(year))
                    temp_config['Flora']['parameters_query']['query'] = query
                    
                    api = FloraAPI(temp_config['Flora'])
                    with api.session():
                        data = api.fetch_data()
                        normalized_data = api.normalize_data(data)
                
                else:  # Scopus
                    api = ScopusAPI(temp_config['Scopus'])
                    normalized_data = api.fetch_data()
                
                # Store data in database if db_ops is provided
                if db_ops:
                    db_ops.store_data_by_source(api_name, normalized_data)
                    logging.info(f"Stored {len(normalized_data)} records from {api_name} for year {year} in database")
                else:
                    # Fallback to CSV if no database operations provided
                    file_path = generate_file_path(api_name, temp_config, year)
                    api.generate_csv_file(file_path, normalized_data)
                    logging.info(f"Saved data for {api_name} to {file_path}")
            
            except Exception as e:
                logging.error(f"Error processing {api_name} for year {year}: {e}")
                continue
        
        # # Process Web of Science
        # try:
        #     temp_config = copy.deepcopy(config)
        #     temp_config['WoS']['PUBLICATION_YEAR'] = year
        #     workflow = WorkflowManager(temp_config['WoS'])
        #     data = workflow.run()
            
        #     # Store WoS data in database if db_ops is provided
        #     if db_ops:
        #         db_ops.store_data_by_source('WoS', data)
        #         logging.info(f"Stored WoS data for year {year} in database")
            
        #     logging.info(f"Completed WoS processing for year {year}")
        # except Exception as e:
        #     logging.error(f"Error processing WoS for year {year}: {e}")
