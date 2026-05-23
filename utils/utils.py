from pathlib import Path
from langchain.prompts import PromptTemplate

def load_prompt(path: str, variables: list[str], encoding):
    text = Path(path).read_text(encoding=encoding)

    return PromptTemplate(
        input_variables=variables,
        template=text
    )