import streamlit as st
from openai import OpenAI


class QnAService:
    """OpenAI API를 이용한 질문/답변 서비스"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)

    def get_answer(self, user_question: str, model: str) -> str:
        """사용자 질문에 대한 해당 모델 답변 생성"""
        try:
            response = self.client.responses.create(
                model=model,
                input=user_question
            )
            return response.output_text
        except Exception as e:
            return f"오류 발생: {str(e)}"


class QnAApp:
    """Streamlit QnA UI 클래스"""
    def __init__(self):
        self.api_key = None
        self.question = None
        self.models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-5-nano", "gpt-5-mini", "gpt-5"]

    def run(self):
        st.header("모델별 응답")
        self.api_key = st.text_input("OPENAI API KEY를 입력하세요.", type="password")
        model = st.selectbox("언어모델을 선택하세요.", options=self.models)
        self.question = st.text_input("질문을 입력하세요.")

        if st.button("답변 확인"):
            if not self.api_key or not self.question:
                st.warning("API 키와 질문을 모두 입력해주세요.")
            else:
                qna_service = QnAService(self.api_key)
                with st.spinner(f"{model} 모델이 답변을 생성 중입니다..."):
                    answer = qna_service.get_answer(self.question, model)
                    st.markdown(f"{model} 답변: {answer}")


if __name__ == "__main__":
    app = QnAApp()
    app.run()