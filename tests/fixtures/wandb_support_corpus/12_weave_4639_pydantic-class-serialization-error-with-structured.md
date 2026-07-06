# Pydantic class serialization error with structured output

- Source: https://github.com/wandb/weave/issues/4639
- Repo: wandb/weave · Issue #4639 · State: closed (closed 2026-02-24)
- Labels: none
- Topic: serialization · Difficulty: med

## Report

We are using LLM calls with structured output, as shown in the dummy example below. Whether we run this with the OpenAI package or Langchain, we get an error saying:

`Error in WeaveTracer.on_chat_model_start callback: TypeError("Object of type 'ModelMetaclass' is not JSON serializable")`

Because the Pydantic class can't be serialized properly in that callback. The only workaround we've found is to basically not use Pydantic objects as structured output definitions and instead convert them to json_schemas, but that undermines the usefulness of that method. 

It does look like Pydantic serialization is handled properly elsewhere in the repo, just not in this callback?

We are evaluating weave vs langsmith, so we could well be using it wrong. 

```
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import weave

weave.init("simple-llm-test")

class Process(BaseModel):
    canonical_process_name: str = Field(..., description="The canonical process")
    name: str = Field(..., description="The company specific alias for the canonical process")

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14").with_structured_output(Process)
result = llm.invoke("give me a process for a company that is called 'Onboarding new data engineers' and the canonical process is 'Employee Onboarding'")
```

Edit: I should note it's an informational error but the code still executes and a partial trace is logged correctly. And if we switch the structured_output definition to json it all works fine, so the rest of the code seems good.

## Discussion / resolution

_(no comments)_
