import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel, ConfigDict, field_validator
from redis import asyncio
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model_path = os.path.join(os.path.dirname(__file__), "../distilbert-base-uncased-finetuned-sst2")
logger.debug(f"Model path: {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    top_k=None,
)

LOCAL_REDIS_URL = "redis://localhost:6379"

@asynccontextmanager
async def lifespan(app: FastAPI):
    HOST_URL = os.environ.get("REDIS_URL", LOCAL_REDIS_URL)
    logger.debug(HOST_URL)
    redis = asyncio.from_url(HOST_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache-project")

    yield


sub_application_sentiment_predict = FastAPI(lifespan=lifespan)


class SentimentRequest(BaseModel):
    model_config = ConfigDict(extra='forbid') # No extra fields
    text: list[str]

    @field_validator("text")
    @classmethod
    def no_empty_strings(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Text list contains an empty or whitespace-only string")
        return value


class Sentiment(BaseModel):
    label: str
    score: float


class SentimentResponse(BaseModel):
    predictions: list[list[Sentiment]]

@sub_application_sentiment_predict.post(
    "/bulk-predict", response_model=SentimentResponse
)
@cache(expire=60)
async def predict(sentiments: SentimentRequest):
    return {"predictions": classifier(sentiments.text)}

@sub_application_sentiment_predict.post(
    "/bulk-predict", response_model=SentimentResponse
)
@cache(expire=60)
async def predict(sentiments: SentimentRequest):
    results = [classifier(text) for text in sentiments.text]
    return SentimentResponse(predictions=results)
    #return {"predictions": classifier(sentiments.text)}

@sub_application_sentiment_predict.post(
        "/bulk-predict", response_model=SentimentResponse
)
@cache(expire=60)
async def predict(sentiments: SentimentRequest):
    results = classifier(sentiments.text)  # List of lists of dicts
    formatted_predictions = [
        [Sentiment(**item) for item in inner_list] for inner_list in results
    ]
    return SentimentResponse(predictions=formatted_predictions)

@sub_application_sentiment_predict.get("/health")
async def health():
    return {"status": "healthy"}
