import asyncio

from sl.llm.data_models import Judgment, LLMResponse, Model, SampleCfg
from sl.llm.data_models import MessageRole, Chat, ChatMessage
from sl.external import openai_driver


def build_simple_chat(user_content: str, system_content: str | None = None) -> Chat:
    if system_content is not None:
        messages = [
            ChatMessage(role=MessageRole.system, content=system_content),
            ChatMessage(role=MessageRole.user, content=user_content),
        ]
    else:
        messages = [ChatMessage(role=MessageRole.user, content=user_content)]
    return Chat(messages=messages)


async def sample(model: Model, input_chat: Chat, sample_cfg: SampleCfg) -> LLMResponse:
    match model.type:
        case "openai":
            return await openai_driver.sample(model.id, input_chat, sample_cfg)
        case "open_source":
            responses = await batch_sample(
                model, [input_chat], [sample_cfg]
            )
            return responses[0]
        case _:
            raise NotImplementedError


async def batch_sample(
    model: Model, input_chats: list[Chat], sample_cfgs: list[SampleCfg]
) -> list[LLMResponse]:
    assert len(input_chats) == len(sample_cfgs)
    match model.type:
        case "openai":
            return await openai_driver.batch_sample(
                model.id, input_chats=input_chats, sample_cfgs=sample_cfgs
            )
        case "open_source":
            from sl.external import transformers_driver  # noqa

            return await asyncio.to_thread(
                transformers_driver.batch_sample,
                model,
                input_chats,
                sample_cfgs,
            )
        case _:
            raise NotImplementedError


async def judge(judgment: Judgment, prompt: str, response: LLMResponse) -> LLMResponse:
    query = judgment.template.format(prompt=prompt, completion=response.completion)

    return await sample(
        judgment.judge_model, build_simple_chat(user_content=query), judgment.sample_cfg
    )


async def batch_judge(
    judgment: Judgment, prompts: list[str], responses: list[LLMResponse]
) -> list[LLMResponse]:
    queries = [
        judgment.template.format(prompt=p, completion=r.completion)
        for (p, r) in zip(prompts, responses)
    ]
    input_chats = [build_simple_chat(q) for q in queries]

    return await batch_sample(
        judgment.judge_model,
        input_chats,
        [judgment.sample_cfg for _ in range(len(queries))],
    )


async def next_token_topk(
    model: Model,
    input_chats: list[Chat],
    top_k: int = 5,
    include_tokens: list[str] | None = None,
) -> list[dict]:
    """
    Compute next-token probabilities for each chat.
    Returns list of dicts (prompt, topk, included).
    """
    match model.type:
        case "open_source":
            from sl.external import transformers_driver  # noqa

            return await asyncio.to_thread(
                transformers_driver.next_token_topk,
                model,
                input_chats,
                top_k,
                include_tokens,
            )
        case "openai":
            raise NotImplementedError("next-token eval not implemented for OpenAI models")
        case _:
            raise NotImplementedError
