from typing import Union
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from query_data import query_rag
from delete_docs import delete_all_docs
from populate_database import popuiate_database
from textract_wrapper import TextractWrapper
import boto3
import aiofiles
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    response: str
    recommendation: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(req: QuestionRequest):
    print("req", req)
    ctx = f"""
     Given the following question: {req.question}
     Use this recommendation answer as base for your answer together with the context: {req.recommendation}
     Tell me if this response is a good answer for the question, it not needs to be the same as your context or the recommendation, but at leat good: *{req.response}*
    
    ---
     
     Give me your answer in the format of JSON. I want a key "motive" with the value being your reasoning and what should be 
     the correct answer. Answer the value of "motive" key in portuguese. I want one more key "percentage" with the value being how close is from the at least good solution. It must be between 0 and 100. I want another 
     key "ideal" with the value being the complete ideal answer that it's present on the context given.
     In your JSON response, everything that it's not the value of "motive", "percentage" and "ideal", don't tell me. The keys "motive" and "percentage" are required and must be in English. 
     The type of all values should only be string.
    """

    prompt_template = ChatPromptTemplate.from_template(ctx)
    prompt = prompt_template.format(question=req.question, response=req.response)
    
    answer = query_rag(prompt)
    return answer

out_file_path = "./data"

@app.delete("/vectors")
def delete_vectors():
    delete_all_docs(out_file_path)
    return {"message": "Deleted all vectors."}


@app.post("/uploadfiles")
async def create_upload_files(files: list[UploadFile]):
    for file in files:
        async with aiofiles.open(f"{out_file_path}/{file.filename}", "wb") as buffer:
            data = await file.read()
            await buffer.write(data)
            
    popuiate_database()
    
    return {"message": "files saved", "filenames": [file.filename for file in files]}


@app.post("/extract")
async def extract_text(file: UploadFile):
    textract_client = boto3.client("textract")
    textract = TextractWrapper(textract_client, None, None)
    document_bytes = await file.read()
    response = textract.detect_file_text(document_bytes=document_bytes)
    blocks = response["Blocks"]
    lines = [block["Text"] for block in blocks if block["BlockType"] == "LINE"]
    text = " ".join(lines)
    return {"text": text}