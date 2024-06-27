from fastapi import FastAPI
from predict import Predict

app = FastAPI()

pre = Predict()
@app.get("/classmc/{text}")
async def classfier(text):
    try:
        print(text)
        return {"classfier":pre.predict(text)}
    except Exception as  e:
        print(e)
        return e
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    pass
