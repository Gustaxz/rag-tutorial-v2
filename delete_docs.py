from langchain.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function
import os
import shutil

CHROMA_PATH = "chroma"

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError as err:
       
     print("Error occurred while deleting files.", err)

def delete_all_docs(filesPath):
    db = Chroma(
    persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    existing_items = db.get(include=[])
    ids = existing_items["ids"]
    
    if not ids:
        return db
    
    db.delete(ids)
    
    delete_files_in_directory(filesPath)
    shutil.rmtree("./chroma", ignore_errors=True, onerror=None)
    
    return db

def main():
    db = delete_all_docs("./app/data")
    print("Deleted all documents from the database. Remaining documents: {}".format(len(db.get(include=[])["ids"])))

if __name__ == "__main__":
    main()
