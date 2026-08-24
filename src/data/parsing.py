"""生成结果解析：从模型吐出的一整段文本里切出 think 和 answer，以及从题面里切出参考资料。

本模块在整条链路里的位置：数据构建、rollout 选样、在线奖励、离线评测——凡是拿到一段模型
输出要看内容的地方，第一步都调这里。

模型的输出长这样：

    <think>
    推理过程
    </think>

    <answer>
    最终答复
    </answer>

两个解析函数是**故意分开**的，不是重复实现：

- :func:`parse_think_answer` 尽量把内容捞出来，标签残缺也不报错。数据构建阶段用它，
  因为那时的输入是贪心生成的完整输出，容错捞取只会多救回几条。
- :func:`parse_think_answer_diagnostic` 会把"没写完"如实标成格式失败。评测和在线奖励用它。

这条区分是有代价的：容错版遇到"生成到最大长度、卡在 think 里就断了"的输出，会把整段
残缺推理当成 answer 交出去。评测里一旦这么干，一条彻底没写完的输出会被算成"答案在池"，
指标就虚高了。所以评测侧一律走 diagnostic 版。
"""

from __future__ import annotations

# 题面里参考资料段的起止标记。语料是这个格式，改了语料就要改这两个常量。
_REF_BEGIN = "【参考问答对】"
_REF_END = "【问题】"


def parse_think_answer(text: str) -> tuple[str, str]:
    """容错解析：尽量捞出 (think, answer)，标签残缺不报错。

    基座模型不靠模板注入开标签，是自己吐完整的 `<think>...</think><answer>...</answer>`，
    所以这里用"找 `</think>`、剩下的当 answer"的写法，对"自己吐"和"模板注入"两种都能解。

    它抓不到什么：生成被最大长度截断、一直卡在 think 里的输出。这种情况下 `</think>` 找不到，
    整段会被当成 answer 返回。评测和在线奖励绝不能用这个函数，用
    :func:`parse_think_answer_diagnostic`。

    :param text: 模型的一整段输出
    :return: (think, answer)，都已 strip
    """
    text = text or ""
    close = text.find("</think>")
    if close != -1:
        think = text[:close].replace("<think>", "").strip()
        tail = text[close + len("</think>"):]
    else:
        think, tail = "", text

    open_answer = tail.find("<answer>")
    if open_answer != -1:
        close_answer = tail.rfind("</answer>")
        inner = (tail[open_answer + len("<answer>"): close_answer]
                 if close_answer > open_answer else tail[open_answer + len("<answer>"):])
        answer = inner.replace("<answer>", "").replace("</answer>", "").strip()
    else:
        answer = tail.replace("<answer>", "").replace("</answer>", "").strip()
    return think, answer


def parse_think_answer_diagnostic(text: str) -> dict:
    """诊断解析：切出 think/answer，并如实标出格式失败。

    与容错版唯一的实质区别在 `</think>` 缺失这一支：这里保留观察到的那段残缺推理，
    answer 留空，并记下失败原因，绝不让容错兜底把残缺推理伪装成答案。

    本函数不生成任何文本，也不从金标准答案里复制任何内容——评测必须只看模型真正写出来的东西。

    :param text: 模型的一整段输出
    :return: ``{think, answer, format_ok, format_reason}``；`format_reason` 用 `+` 连接多个原因
    """
    raw = text or ""
    close = raw.find("</think>")
    if close != -1:
        think = raw[:close].replace("<think>", "").strip()
        tail = raw[close + len("</think>"):]
        open_answer = tail.find("<answer>")
        if open_answer != -1:
            close_answer = tail.rfind("</answer>")
            inner = (tail[open_answer + len("<answer>"): close_answer]
                     if close_answer > open_answer else tail[open_answer + len("<answer>"):])
            answer = inner.replace("<answer>", "").replace("</answer>", "").strip()
        else:
            answer = tail.replace("<answer>", "").replace("</answer>", "").strip()
    else:
        # 子分支：输出从头到尾没离开推理段。保留这段观察到的残缺推理，answer 明确置空。
        think = raw.replace("<think>", "", 1).strip() if "<think>" in raw else ""
        answer = ""

    reasons = []
    if close == -1:
        reasons.append("missing_think_close")
    if not think:
        reasons.append("empty_think")
    if not answer:
        reasons.append("empty_answer")
    return {
        "think": think,
        "answer": answer,
        "format_ok": not reasons,
        "format_reason": "+".join(reasons) if reasons else "ok",
    }


def extract_references(user_prompt: str) -> str:
    """从题面里切出【参考问答对】和【问题】之间的参考资料。

    裁判判"这段 think 是不是在复述参考资料"，需要拿到参考资料原文。整个题面直接喂给裁判
    不行——题面尾部还带着问题本身，会让裁判把"复述题干"也算成照抄。

    :param user_prompt: 完整题面
    :return: 参考资料段；题面里没有起始标记就返回空串
    """
    prompt = user_prompt or ""
    begin = prompt.find(_REF_BEGIN)
    if begin == -1:
        return ""
    start = begin + len(_REF_BEGIN)
    end = prompt.find(_REF_END, start)
    return prompt[start:end].strip() if end > start else prompt[start:].strip()


def wrap_completion(think: str, answer: str) -> str:
    """按训练格式把 think 和 answer 拼成一整段 assistant 内容。

    开头那个 `<think>` 必须带上：模型是自己吐标签的，训练目标里少了它，学出来的模型
    就不吐开标签，下游所有解析全部失效。

    :param think: 推理段
    :param answer: 答复段
    :return: 拼好的完整输出文本
    """
    return f"<think>\n{(think or '').strip()}\n</think>\n\n<answer>\n{(answer or '').strip()}\n</answer>"
