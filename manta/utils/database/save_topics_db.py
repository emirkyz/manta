from sqlalchemy import create_engine, MetaData, Table, Column, String, insert, text

def save_topics_to_db(topics_data, data_frame_name, topics_db_eng):
    """Save topics to database using SQLAlchemy engine.

    Args:
        topics_data (dict): A dictionary containing topic names as keys and lists of words as values.
        data_frame_name (str): The name of the data frame.
        topics_db_eng (sqlalchemy.engine.Engine): The SQLAlchemy engine to use for the database.

    Raises:
        ex: Exception
    """
    if not topics_db_eng:
        print("Warning: No database engine provided, skipping database save")
        return
    try:
        # Create metadata
        metadata = MetaData()
        
        table_name = f"{data_frame_name}_topics"
        
        # Drop table if it exists
        with topics_db_eng.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            
        # Create table with dynamic columns based on topics
        columns = [Column(topic, String) for topic in topics_data.keys()]
        table = Table(table_name, metadata, *columns)
        
        # Create table
        metadata.create_all(topics_db_eng)

        # Prepare data for insertion
        max_words = max(len(words) for words in topics_data.values())
        rows = []
        for i in range(max_words):
            row = {}
            for topic, words in topics_data.items():
                row[topic] = words[i] if i < len(words) else None
            rows.append(row)

        # Insert data using SQLAlchemy
        with topics_db_eng.begin() as conn:
            if rows:  # Only insert if we have data
                conn.execute(insert(table), rows)
                
    except Exception as ex:
        print(f"Error saving topics to database: {ex}")
        raise ex
