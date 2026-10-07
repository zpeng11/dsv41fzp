---
id: P0000
title: <一句话意图>
status: draft        # draft | active | completed | abandoned
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>
knowledge: pending  # pending | updated | covered | none | deferred
docs: []            # 长期事实、契约或操作知识的归档或依据；可带 #章节
bench: []           # 关联 bench/results/<日期-主题>
decisions: []       # 关联 D 编号
followups: []       # 接收剩余义务的 P 编号，不得自指
---

## 意图

<为什么做、成功长什么样。不写实现细节。>

## 验收

- [ ] <可判定的验收条件；性能类条件须注明 bench 目录与对比基线（引擎版本 + 既有 bench 目录）>

## 记录

<过程要点与证据指针，不复制长期知识正文。尚未产生的归档目标在此或意图中说明，不在 front matter 填入悬空引用。>

## 收尾

- 验收或停止依据：<完成时引用验收证据并明确未验证部分；放弃时说明停止原因和已有结果。>
- 知识处置：<updated：知识已新增或修订；covered：既有材料完整覆盖；none：具体说明为何无长期知识增量；deferred：说明移交内容及为何不阻塞交付。updated / covered 至少回填一项 docs / bench / decisions；pending 不得进入终态。>
- 剩余义务：<无，或说明内容与接收 P 编号并回填 followups。必要知识缺失仍阻塞交付，不得以 deferred 结项。>
