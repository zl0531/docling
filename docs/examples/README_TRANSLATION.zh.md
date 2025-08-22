# 使用本地 LLM 和 Argos Translate 翻译文档

这个示例演示了如何使用翻译服务将文档文本翻译并生成 HTML 文件。

## 文件列表

所有 Python 文件都已移动到项目根目录以便于导入：

- `convert_to_interactive_html.py`：用于转换文档并可选择翻译的主要脚本
- `local_llm.py`：用于调用本地 LLM 服务的 LocalLLM 类实现
- `argos_translate.py`：用于离线翻译的 ArgosTranslate 类实现
- `test_local_llm.py`：LocalLLM 类的测试脚本
- `test_argos_translate.py`：ArgosTranslate 类的测试脚本
- `.env.example`：环境变量模板（位于项目根目录）
- `LOCAL_LLM_USAGE.md`：使用本地 LLM 模型的文档（在 docs/examples/ 目录中）

## 设置

1. 从项目根目录复制 `.env.example` 文件为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 在 `.env` 文件中填入你的实际值：
   ```bash
   LOCAL_LLM_APP_KEY=your_app_key_here
   LOCAL_LLM_SECRET_KEY=your_secret_key_here
   LOCAL_LLM_APP_CODE=your_app_code_here
   ```

3. 对于 Argos Translate，安装包：
   ```bash
   pip install argostranslate
   ```

## 使用方法

转换文档并使用本地 LLM 进行翻译：
```bash
python convert_to_interactive_html.py path/to/document.pdf --translate --translation-backend local_llm
```

转换文档并使用 Argos Translate（离线）进行翻译：
```bash
python convert_to_interactive_html.py path/to/document.pdf --translate --translation-backend argos
```

只转换文档不翻译：
```bash
python convert_to_interactive_html.py path/to/document.pdf
```

测试 LocalLLM 类：
```bash
python test_local_llm.py
```

测试 ArgosTranslate 类：
```bash
python test_argos_translate.py
```

## 工作原理

1. `convert_to_interactive_html.py` 脚本使用 Docling 将文档转换为各种格式
2. 当使用 `--translate` 参数时，会调用指定的翻译后端来翻译文本
3. 同时保存原文和译文版本

## 翻译后端

### 本地 LLM
- 需要访问本地 LLM 服务
- 提供高质量翻译
- 需要网络连接
- 需要在 `.env` 文件中配置认证凭据

### Argos Translate
- 使用机器学习模型进行离线翻译
- 初始设置后无需网络连接
- 自动下载和安装翻译模型
- 对常见语言对提供良好质量

## 本地 LLM 实现

`local_llm.py` 中的 `LocalLLM` 类负责：
- 与本地 LLM 服务进行认证
- 管理和缓存访问令牌
- 向 LLM 服务发送翻译请求
- 处理错误情况

更多关于使用本地 LLM 模型的详细信息，请参阅 `docs/examples/LOCAL_LLM_USAGE.md`。

## Argos Translate 实现

`argos_translate.py` 中的 `ArgosTranslate` 类负责：
- 自动下载和安装翻译模型
- 使用机器学习进行离线翻译
- 支持多种语言对

## 故障排除

如果遇到问题：

1. **环境变量未加载**：确保 `.env` 文件位于项目根目录并且包含正确的值。

2. **令牌获取失败**：检查 `LOCAL_LLM_APP_KEY` 和 `LOCAL_LLM_SECRET_KEY` 是否正确。

3. **网络问题**：验证您可以访问本地 LLM 服务：
   - 办公网络：`https://ea-ai-gateway.corp.kuaishou.com/ea-ai-gateway/open/v1`
   - IDC：`http://ea-ai-gateway.common.ee-prod.internal:18080/ea-ai-gateway/open/v1`

4. **Argos Translate 问题**：
   - 确保您有足够的磁盘空间来下载翻译模型。
   - 如果出现 "No module named 'argostranslate'" 错误，请确保已使用 `pip install argostranslate` 安装
   - 首次翻译可能需要一些时间，因为它需要下载模型

5. **调试**：您可以运行测试脚本以获取更多日志信息：
   ```bash
   python test_local_llm.py
   python test_argos_translate.py
   ```

6. **TableCell 属性错误**：如果您遇到关于 `TableCell` 没有 `orig` 属性的错误，这是正常的，因为我们无法修改 `TableCell` 对象的结构。当前实现会在不存储原始文本的情况下翻译表格单元格文本。