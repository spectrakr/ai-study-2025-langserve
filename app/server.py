from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes

from chains.chains import ChatChain, TopicChain, LLM, Translator
from chains.rag import RagChain

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/chat/playground")


add_routes(app, Translator().create(), path="/translate")
add_routes(app, LLM().create(), path="/llm")
add_routes(app, TopicChain().create(), path="/topic")
add_routes(
    app,
    RagChain(file_path="data/SPRI_AI_Brief_2023년12월호_F.pdf").create(),
    path="/rag",
)

########### 대화형 인터페이스 ###########


class InputChat(BaseModel):
    """채팅 입력을 위한 기본 모델 정의"""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


# 대화형 채팅 엔드포인트 설정
# LangSmith를 사용하는 경우, 경로에 enable_feedback_endpoint=True 을 설정하여 각 메시지 뒤에 엄지척 버튼을 활성화하고
# enable_public_trace_link_endpoint=True 을 설정하여 실행에 대한 공개 추적을 생성하는 버튼을 추가할 수도 있다.
# LangSmith 관련 환경 변수를 설정해야 한다(.env)
add_routes(
    app,
    ChatChain().create().with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)


# 서버 실행 설정
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
