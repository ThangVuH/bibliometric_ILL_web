import json
import logging
from datetime import datetime
import pandas as pd
from quart import Quart, request, jsonify
from quart_cors import cors

import fetch_data
from database_manager import CitationNetworkDatabase, DataProcessor, DatabaseConfig
from citation_network import CitationNetworkBuilder
from patent_network import PatentPipeline, Config

# CONFIG_FILE = r"C:\Users\vu-hong\Desktop\davies\backend_2\config.json"
CONFIG_FILE = r"C:\Users\vu-hong\Desktop\davies\web_biblio\backend\config.json"

app = Quart(__name__)
app = cors(app, allow_origin="*")

@app.route('/')
def home():
    return "<h1>Bibliometric data</h1><p>Data is being fetched, stored and analysed.</p>"


@app.route('/fetch_data', methods=['GET'])
def fetch_data_endpoint():
    """
    Endpoint to trigger data fetching and storing into 4 databases: 
    [flora_publications, openalex_publications, scopus_publications, wos_publications].
    """
    try:
        # Get parameters from query string with defaults
        start_year = request.args.get('start_year', default=1970, type=int)
        end_year = request.args.get('end_year', default=datetime.now().year, type=int)

        # Load configuration
        config = fetch_data.load_config(CONFIG_FILE)

        # Create database session
        db_manager = CitationNetworkDatabase(CONFIG_FILE)
        db_manager.create_tables()

        try:
            # Fetch data and store directly in database
            fetch_data.extract_data_for_years(config, start_year, end_year, db_manager)

            # Log success
            app.logger.info(f"Successfully fetched and stored data for years {start_year}-{end_year}")
            
            return jsonify({
                'status': 'success',
                'message': f'Data fetched and stored successfully for years {start_year}-{end_year}',
                'years_processed': {'start': start_year, 'end': end_year}
            })
        
        except Exception as e:
            raise e
    except Exception as e:
        # Log the error for server-side debugging
        app.logger.error(f"Error during data fetch: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    
@app.route('/validate_data')
def validate_data():
    """
    Preprocessing data and save to 'preprocess_doi' database.
    Includes tier classification based on Flora document types.
    Can filter by year range using query parameters start_year and end_year.
    """
    try:
        # Load configuration
        config = DatabaseConfig(CONFIG_FILE).config
        db_manager = CitationNetworkDatabase(CONFIG_FILE)
        data_processor = DataProcessor(db_manager)

        # Step 2: Process data
        processed_count = data_processor.process_dois()
        print(f"Processed {processed_count} DOIs")

        # Step 4: Get statistics
        stats = db_manager.get_statistics()

        return jsonify({
            'status': 'success',
            'message': 'Data validation completed',
            "Database Statistics:": stats
        })

    except Exception as e:
        app.logger.error(f"Error during data validation: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    
@app.route('/citation_data')
async def citation_data():
    try:
        # Load configuration
        config = DatabaseConfig(CONFIG_FILE).config
        db_manager = CitationNetworkDatabase(CONFIG_FILE)

        # Initialize the builder
        builder = CitationNetworkBuilder(
            db_config=CONFIG_FILE,
            email="vu-hong@ill.fr",
            api_key=None  # Optional: add your OpenAlex API key
        )
        d0_success, d0_failure = await builder.build_d0_works_from_processed_dois(tier = None)

        d0_processed = builder.process_d0_citations()

        d1_success, d1_failure, expected_citations, actual_citations = await builder.build_d1_citations(work_filter = "cited_by_count > 0")

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
        return jsonify({
            'status': 'success',
            'message': 'Build Citation network completed',
            "Database Statistics:": (stats)
        })
    
    except Exception as e:
        app.logger.error(f"Error during citation network: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
    
@app.route('/patent_data')
def patent_data():
    try:
        config = fetch_data.load_config(CONFIG_FILE)
        config = Config(config["Lens"])
        pipeline = PatentPipeline(config)
        
        pipeline.run_full_pipeline()
        # pipeline.run_analysis_only()

        return jsonify({
            'status': 'success',
            'message': 'Patents analysis completed'
        })
    except Exception as e:
        app.logger.error(f"Error during analysis patent: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500



if __name__ == '__main__':
    app.run(debug=True)