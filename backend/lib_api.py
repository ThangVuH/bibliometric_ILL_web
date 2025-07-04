import requests
from urllib.parse import urlparse
import json
import xmltodict
import csv
import threading
import os
import re
import copy
from contextlib import contextmanager

class LibraryAPI:
    def __init__(self, config):
        self.config = config
        self.base_url = self.config['URL']

    def fetch_data(self):
        """Placeholder for fetch data; should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def normalize_data(self, data):
        """Placeholder for normalizing data; should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def run_query(self, url, params, timeout=50):
        """Send a GET request to the provided URL with the given parameters."""
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors
            xml_data = response.text
            json_data = json.dumps(xmltodict.parse(xml_data), indent=2)
            json_data = json.loads(json_data)
            return json_data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
        
    def clean_doi(self, doi):
        """Normalize DOIs by stripping prefixes and standardizing format."""
        if doi is None:
            return None
        doi = doi.strip().lower()
        doi = re.sub(r'^(https?://)?(dx\.)?doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)
        for prefix in ['https://doi.org/', 'http://doi.org/', 'doi.org/', 'doi:']:
            if doi.startswith(prefix):
                doi = doi.replace(prefix, '')
                break
        return doi
    
    def generate_csv_file(self, file_name, data):
        if not data:
            raise ValueError("The data list is empty. Cannot generate CSV file.")
        try:
            with open(file_name, mode='w', newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(data[0].keys())  # Write header
                for item in data:
                    writer.writerow(item.values())  # Write rows
            print(f"File '{file_name}' has been created successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
    

class FloraAPI(LibraryAPI):
    def __init__(self, config):
        super().__init__(config)
        self.params_query = self.config['parameters_query']
        self.params_record = self.config['parameters_record']
        self.session_id = None
        self.batch_size = 200
        
        publication_year = config["PUBLICATION_YEAR"]
        query_template = self.params_query["query"]
        query = query_template.replace("{year}", str(publication_year))
        self.params_query["query"] = query

    @contextmanager
    def session(self):
        """Manage Flora API session."""
        try:
            self.login()
            yield
        finally:
            self.logout()

    def login(self):
        USER = self.config['USER']
        PASSWORD = self.config['PASSWORD']
        login_url = f'{self.base_url}?method=login&code={USER}&password={PASSWORD}'
        response = requests.get(login_url)
        if response.ok:
            self.session_id = response.text.split('apiSession>')[1].split('</')[0]
            # print(f"SESSION_ID: {self.session_id}")
            return self.session_id
        else:
            print("Login failed.")
            return None
        
    def logout(self):
        if self.session_id:
            logout_url = f'{self.base_url}?method=logout&apiSession={self.session_id}'
            requests.get(logout_url)
            print("Logged out.")
        else:
            print("No session to log out.")

    def fetch_data(self):
        # Ensure we are logged in before fetching data
        if not self.session_id:
            self.session_id = self.login()
            if not self.session_id:
                print("Unable to login, aborting fetch_data.")
                return {}
        # Step 1: Fetch all IDs
        record_ids = self.fetch_ids()
        if not record_ids:
            print("No IDs fetched.")
            return {}
        print(f"Total results from Flora: {len(record_ids)}")
        
        # Step 2: Fetch detailed records
        records = self.fetch_records_in_batches(record_ids)
        return records
    
    def fetch_ids(self):
        try:
            url = f"{self.base_url}?apiSession={self.session_id}"
            data = self.run_query(url, self.params_query)
            ids = [item['@recordId'] for item in data['response']['digests']['digest']]
            return ids
        except requests.exceptions.RequestException as e:
            print(f"Error fetching IDs: {e}")
            return []
    
    def fetch_records_in_batches(self, record_ids):
        batch_size= self.batch_size
        all_records = []
        for i in range(0, len(record_ids), batch_size):
            batch = record_ids[i:i + batch_size]
            record_id_params = "&".join([f"recordId={record_id}" for record_id in batch])
            self.session_id = self.login()
            url = f"{self.base_url}?apiSession={self.session_id}&{record_id_params}"
            try:
                data = self.run_query(url, self.params_record)
                all_records.append(data)
            except requests.exceptions.RequestException as e:
                print(f"Error fetching batch {i // batch_size + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error fetching batch {i // batch_size + 1}: {e}")
                # return []
        return all_records
    
    def normalize_data(self, data):
        normalize_data = []
        for item in data:
            infos = item['response']['records']['record']
            for info in infos:
                # id here
                if info.get('DIGEST_NUMBER') == None:
                    id = info.get('@id')
                else:
                    id = info.get('DIGEST_NUMBER')
                # doi here
                doi = info.get('CHAMP3')

                # title here
                title = info.get('DIGEST_TITLE')

                # journal title
                source = info.get('DIGEST_JRNAL_TITLE')

                # publish year here
                year = info.get('DIGEST_YEAR')

                # publish year here
                if info.get('DIGEST_YEAR') == None:
                    year = None
                else:
                    year = info.get('DIGEST_YEAR')

                normalize_data.append({
                    'id':id,
                    'doi': self.clean_doi(doi),
                    'title': title,
                    'source': source,
                    'year': year
                })
        return normalize_data
    
class OpenAlexAPI(LibraryAPI):
    def __init__(self, config):
        super().__init__(config)
        self.institution_id = self.config['ROR']
        self.year = self.config['PUBLICATION_YEAR']
        self.per_page = self.config['PER_PAGE']

    def fetch_single_query(self, params, results, lock):
        """
        Fetch data for a single query with cursor pagination.
        """
        cursor = "*"
        while cursor:
            params["cursor"] = cursor
            response = requests.get(self.base_url, params=params)
            if response.status_code != 200:
                print(f"Failed to fetch data for {params}")
                return
            data = response.json()
            this_page_results = data.get("results", [])
            lock.acquire()  # Ensure thread-safe appending to the shared list
            results.extend(this_page_results)
            lock.release()
            cursor = data["meta"].get("next_cursor")

    def fetch_data(self,queries):
        """
        Fetch data for multiple queries using multithreading.
        queries: List of parameter dictionaries for each query.
        """
        threads = []
        results = []
        lock = threading.Lock()  # Lock to manage shared resource (results list)
        # Create and start a thread for each query
        for params in queries:
            thread = threading.Thread(target=self.fetch_single_query, args=(params, results, lock))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        print(f"Total results: {len(results)}")
        return results
    
    def normalize_data(self, data):
        return [
            {
                'id': f"{urlparse(item.get('id')).netloc}{urlparse(item.get('id')).path}",
                'doi': self.clean_doi(f"{urlparse(item.get('doi')).netloc}{urlparse(item.get('doi')).path}")if item.get('doi') else None,
                'title': item.get('title'),
                'type': item.get('type'),
                'source': f"{item['primary_location']['source']['issn_l']} | {item['primary_location']['source']['display_name']}" if item['primary_location'] and item['primary_location']['source'] else None,
                'cited_count': item.get('cited_by_count'),
                'year': item.get('publication_year')
            }
            for item in data
        ]

class HalAPI(LibraryAPI):
    def __init__(self, config):
        super().__init__(config)
        self.query = self.config['query']
        self.year = self.config['PUBLICATION_YEAR']
        self.rows = self.config['row']
        self.wt = self.config['write_type']
        self.sort = self.config['sort']

    def fetch_data(self):    
        cursor_mark = "*"
        has_more = True

        all_results = []
        while has_more:
            params = {
                'q': self.query,
                'rows': self.rows,
                'cursorMark': cursor_mark,
                'wt': self.wt,
                'fq': f'submittedDateY_i:{self.year}',
                'sort': self.sort 
            }
            response = requests.get(self.base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                for doc in data['response']['docs']:
                    doc['metadata_url'] = doc['uri_s'] + '/metadata'
                    xml_data = requests.get(doc['metadata_url'])
                    json_data = json.loads(json.dumps(xmltodict.parse(xml_data.content), indent=2))

                    all_results.append(json_data)


                next_cursor_mark = data['nextCursorMark']
                if cursor_mark == next_cursor_mark:
                    has_more = False
                else:
                    cursor_mark = next_cursor_mark
        print(f"Total results from Archives Ouvertes: {data['response']['numFound']}")
        return all_results
    
    def normalize_data(self, data):
        id_keys = ['TEI', 'text', 'body', 'listBibl', 'biblFull', 'publicationStmt', 'idno']
        title_keys = ['TEI', 'text', 'body', 'listBibl', 'biblFull', 'titleStmt', 'title']
        type_keys = ['TEI', 'text', 'body', 'listBibl', 'biblFull', 'profileDesc', 'textClass', 'classCode']
        date_keys = ['TEI', 'text', 'body', 'listBibl', 'biblFull', 'editionStmt', 'edition', 'date']

        normalize_data = []
        for info in data:
            # id here
            id_data = self.get_nested_value(info, id_keys, default="No id")
            for entry in id_data:
                if entry.get('@type') == 'halUri':
                    id = entry.get('#text')
            
            # title here
            title_data = self.get_nested_value(info, title_keys, default={})
            if isinstance(title_data, list):
                title = title_data[0].get('#text', "No Title")
            elif isinstance(title_data, dict):
                title = title_data.get('#text', "No Title")
            else:
                title = "No Title"

            # docType here
            type_data = self.get_nested_value(info, type_keys, default="No type")
            for entry in type_data:
                if entry.get('@scheme') == 'halTypology':
                    type = entry.get('#text')

            # publish year here
            date_data = self.get_nested_value(info, date_keys, default=[])
            year = "No year"
            for entry in date_data:
                if entry.get('@type') == 'whenReleased':
                    year = entry.get('#text').split('-')[0]  # Extract the year part
                    break
            
            normalize_data.append({
                'id': f"{urlparse(id).netloc}{urlparse(id).path}",
                'title': title,
                'type': type,
                'source': id.split('/')[-2],
                'year': year
            })

        return normalize_data
    
    @staticmethod
    def get_nested_value(d, keys, default=None):
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, default)
            elif isinstance(d, list) and isinstance(key, int) and key < len(d):
                d = d[key]
            else:
                print(f"Key '{key}' not found in: {d}")
                return default
        return d
   
class ScopusAPI(LibraryAPI):
    def __init__(self, config):
        super().__init__(config)
        self.affil = self.config['AFFILIATION']
        self.year = self.config['PUBLICATION_YEAR']
        self.query = f'(AFFIL({self.affil}) OR ALL({self.affil})) AND PUBYEAR = {self.year}'
        self.api_key = self.config['API_KEY']
        self.headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }

    def fetch_data(self, max_results=None):
        publications = []
        params = {
            "query": self.query,
            "count": 25,
            "start": 0
        }
        while True:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            if response.status_code != 200:
                print(f"Error: {response.status_code}, {response.text}")
                break
            data = response.json()
            if "search-results" in data and "entry" in data["search-results"]:
                entries = data["search-results"]["entry"]
                if not entries or (len(entries) == 1 and "error" in entries[0]):
                    print("No more entries found or an error occurred.")
                    break
                
                publication_batch = self.normalize_data(entries)
                publications.extend(publication_batch)
                if max_results and len(publications) >= max_results:
                    publications = publications[:max_results]
                    break
            else:
                print("No more entries found.")
                break
            total_results = int(data["search-results"]["opensearch:totalResults"])
            current_start = params["start"] + params["count"]
            
            # Break if we've reached the end or if there's no 'next' link
            if current_start >= total_results or "next" not in [link["@ref"] for link in data["search-results"]["link"]]:
                break
            
            # Move to the next page
            params["start"] = current_start
        return publications
            # return super().fetch_data()

    def normalize_data(self, data):
        publications = []
        for entry in data:
            publication = {
                "id": entry.get("dc:identifier", "No ID").split(":")[-1],  # Extracts Scopus ID
                "doi": entry.get("prism:doi", "No DOI"),
                "title": entry.get("dc:title", "No Title"),
                "type": entry.get("subtypeDescription", "No Type"),  # E.g., Article, Conference Paper
                "source": entry.get("prism:publicationName", "No Source"),
                "cited_count": entry.get("citedby-count", 0),
                "year": entry.get("prism:coverDate", "No Year").split("-")[0]  # Extracts year
            }
            publications.append(publication)
        return publications
        


def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in '{config_path}'.")

def generate_file_path(api_name, config, year):
    """Generate file path for CSV output."""
    download_folder = config[api_name]["download_folder"]
    # file_name = f"{api_name}_{config[api_name]['PUBLICATION_YEAR']}.csv"
    file_name = f"{api_name}_{year}.csv"
    return os.path.join(download_folder, file_name)

def extract_data_for_years(config, start_year, end_year):
    """Extract data for a range of years."""
    for year in range(start_year, end_year + 1):
        temp_config = copy.deepcopy(config)  
        temp_config['Flora']['PUBLICATION_YEAR'] = year  

        # Update the query with the current year
        query_template = temp_config['Flora']['parameters_query']['query']
        query = query_template.replace("{year}", str(year))
        temp_config['Flora']['parameters_query']['query'] = query

        api = FloraAPI(temp_config['Flora'])

        with api.session():
            data = api.fetch_data()
            normalized_data = api.normalize_data(data)
            # print(f"Fetching data for year: {year}")
            print(f"Query being sent: {api.params_query['query']}")

        file_path = generate_file_path('Flora', config, year)
        api.generate_csv_file(file_path, normalized_data)
