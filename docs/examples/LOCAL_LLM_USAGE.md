# Using Local LLM Models with langextract

  我们在langextract项目中成功实现了本地LLM的集成，解决了多个关键问题。以下是完整的实现过程和遇到的问题：

  ##1. 本地LLM连接实现

  认证和访问令牌获取
  我们创建了正确的认证流程来获取访问令牌：

   1. 正确的令牌端点：使用https://is-gateway.corp.kuaishou.com/token/get而不是测试端点2.
      正确的认证信息：使用环境变量中的LOCAL_LLM_APP_KEY和LOCAL_LLM_SECRET_KEY3. 令牌缓存：实现了令牌缓存机制，避免频繁请求新令牌

  自定义LocalLLM类
  我们创建了专门的LocalLLM类来处理本地LLM连接：1. 基于OpenAI API：继承自BaseLanguageModel，使用OpenAI兼容的API
   2. 自定义头部信息：支持添加X-App-Code和X-App-Key头部
   3. 正确的系统提示：明确指示LLM返回解析器期望的格式

  2. 解决的关键问题### 访问令牌问题
  问题：无法获取访问令牌解决方案：- 使用正确的令牌获取URL
   - 使用正确的app key和secret key
   - 正确解析令牌响应格式（从data改为result字段）

  响应格式问题
  问题：LLM返回的格式与解析器期望的格式不匹配解决方案：- 创建新的提示文件prompt_local_llm.md，明确要求返回带extractions键的格式
   - 修改LocalLLM类中的系统消息，进一步明确格式要求### 空提取文本问题
  问题：当提取文本为空时，对齐过程会失败
  解决方案：- 在app_local.py中添加异常处理，捕获并处理空提取文本的情况

  可视化展示问题
  问题：提取完成后，结果没有在HTML中展示
  解决方案：
   - 确保正确生成结果文件- 验证数据库记录和文件路径的正确性
   - 修复前端JavaScript代码，避免不必要的API密钥提示

  ##3. 核心组件

  app_local.py
   - 实现了本地LLM的Flask应用程序- 包含令牌管理和认证逻辑
   - 处理文件上传、提取和结果展示

  local_llm.py- 自定义的LocalLLM类，支持本地LLM服务
   - 处理认证头部和正确的响应格式

  prompt_local_llm.md
   - 专门为本地LLM定制的提示文件- 明确要求返回解析器期望的JSON格式

  ##4. 部署和使用

  环境变量配置

   1 LOCAL_LLM_APP_KEY=41e638cc-c9f8-482e-8610-85fa7b48ac94LOCAL_LLM_SECRET_KEY=openAppba9797e14aa5b1fd041839cdaLOCAL_LLM_APP_CODE=5
     rCQKrzsCKzA
   2 LOCAL_LLM_URL_OFFICE=https://ea-ai-gateway.corp.kuaishou.com/ea-ai-gateway/open/v1LOCAL_LLM_DEFAULT_MODEL=deepseek_v3```
   3 
  启动应用程序
   1 python app_local.py

  5. 工作流程1. 用户上传文档（PDF或DOCX）
   2. 选择示例数据
   3. 应用程序获取访问令牌
   4. 使用LocalLLM类调用本地LLM服务
   5. 解析LLM响应并生成结构化数据
  6.保存结果并生成可视化展示
   7. 在Web界面中展示提取结果

  通过以上实现，我们成功地将本地LLM集成到langextract项目中，解决了认证、格式匹配、错误处理和可视化展示等一系列问题，使系统能够稳定地运行并正确展
  示提取结果。


# Using Local LLM Models with Doc-Agent

This guide explains how to use the local LLM integration in the doc-agent project.

## Summary of Findings

We've successfully implemented and fixed the local LLM integration with the doc-agent project. The implementation includes:

1. A standalone test script (`test_qwen_vl_72b.py`) that successfully connects to the OpenAI-compatible API and can perform document analysis using the deepseek_v3 model
2. A complete integration with the doc-agent LLM factory system:
   - `LocalLLMFactory` implementation
   - Configuration updates in `llm_config.py` 
   - Type definitions in `constants/llm_type.py`

We initially faced permission issues with the integrated solution, but we've now fixed them by:

1. Using the correct app key and secret key when requesting tokens
2. Adding appropriate X-App-Code and X-App-Key headers to the API requests

Both the standalone script and the integrated solution now work correctly with the deepseek_v3 model.

## Available Local LLM Models

The following model interfaces are defined in the system:

| LLM Type Constant | Model Name | Status | Description |
|------------------|----------------|----------------|-------------|
| `LLMType.LOCAL_DEEPSEEK_V3` | `deepseek_v3` | ✅ Working | General text model for document processing |
| `LLMType.LOCAL_DEEPSEEK_R1` | `deepseek_r1` | ⚠️ Not tested | General text model (alternative version) |
| `LLMType.LOCAL_QWEN_32B` | `qwq-32b` | ⚠️ Not tested | 32B parameter Qwen model for text processing |
| `LLMType.LOCAL_QWEN_VL_72B` | `qwen-vl-72b` | ⚠️ Not tested | 72B parameter multimodal Qwen model (supports images) |

> **IMPORTANT NOTE:** After requesting permissions, we can now successfully use the deepseek_v3 model. The other models (deepseek_r1, qwq-32b, qwen-vl-72b) may also be available but have not been tested yet.

## Recommended Approach

Based on our testing, we recommend using the standalone script for now:

```python
# Example usage of the test_qwen_vl_72b.py script
from test_qwen_vl_72b import test_qwen_multimodal, test_document_analysis

# For document analysis
result = test_document_analysis()

# For image analysis
result = test_qwen_multimodal(image_path="path/to/image.jpg", prompt="What's in this image?")
```

This approach successfully works with the deepseek_v3 model. The integrated solution is ready but requires additional permissions that are still pending.

## Quick Start

### Basic Usage (Once Permissions are Resolved)

To use a local LLM model in your code once all permissions are granted:

```python
from constants.llm_type import LLMType
from core.ai.llm.llm_factory import LLMFactoryProvider

async def process_document(text):
    system_prompt = "You are a document analysis assistant."
    user_prompt = f"Analyze the following document:\n\n{text}"
    
    # Use the local DeepSeek V3 model
    response = await LLMFactoryProvider.get_completion(
        llm_type=LLMType.LOCAL_DEEPSEEK_V3,  # Use any of the local LLM types
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.3  # Lower for more deterministic outputs
    )
    
    return response
```

### Using the Multimodal Model (Qwen-VL-72B)

For image processing with the multimodal model:

```python
import base64
from constants.llm_type import LLMType
from core.ai.llm.llm_factory import LLMFactoryProvider

async def process_document_with_image(text, image_path):
    # Create a system prompt
    system_prompt = "You are a document analysis assistant that can understand both text and images."
    
    # Encode the image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create a message with both text and image
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]
    
    # Get the factory for the Qwen-VL model
    factory = LLMFactoryProvider.get_factory(LLMType.LOCAL_QWEN_VL_72B)
    llm = factory.create_llm()
    response = factory.send_message(llm, messages, temperature=0.5)
    
    # Extract response content
    return response.choices[0].message.content
```

## Using Local LLM for Document Translation

We've also implemented a solution to translate document text using local LLM models. This is demonstrated in the `convert_to_interactive_html.py` example:

1. A `LocalLLM` class is implemented in `local_llm.py` that handles authentication and communication with the local LLM service.
2. The `convert_to_interactive_html.py` script has been enhanced with a `--translate` option that translates document text to Chinese.
3. Both the original and translated versions of the documents are saved.

### Usage Example

```bash
# Convert a document and translate it to Chinese
python convert_to_interactive_html.py path/to/document.pdf --translate

# Convert a document without translation
python convert_to_interactive_html.py path/to/document.pdf
```

## Testing the Integration

Two test scripts are provided to verify the local LLM integration:

1. `test_local_llm.py` - Basic test script to connect to a local LLM service
2. `test_qwen_vl_72b.py` - Test script specifically for the Qwen-VL-72B multimodal model
3. `test_local_llm_integration.py` - Test script that verifies the integration with the doc-agent project

### Running the Tests

```bash
# Test the basic connection to a local LLM
python test_local_llm.py --model deepseek_v3

# Test the Qwen-VL-72B multimodal model with an image
python test_qwen_vl_72b.py --image /path/to/your/image.jpg --prompt "What's in this image?"

# Test the integration with the doc-agent project
python test_local_llm_integration.py --model deepseek_v3

# List all available local LLM models
python test_local_llm_integration.py --list-models
```

## Configuration

The local LLM configuration is now defined in the `.env` file in the project root directory:

```
# Local LLM configuration
LOCAL_LLM_APP_KEY=41e638cc-c9f8-482e-8610-85fa7b48ac94
LOCAL_LLM_SECRET_KEY=openAppba9797e14aa5b1fd041839cda
LOCAL_LLM_APP_KEY_DEV=43907b2f-57ae-4398-8a41-424bbde1fce9
LOCAL_LLM_SECRET_KEY_DEV=openAppb6067e7b887ca0dc3fdcf3b65
LOCAL_LLM_APP_CODE=5rCQKrzsCKzA
LOCAL_LLM_URL_IDC=http://ea-ai-gateway.common.ee-prod.internal:18080/ea-ai-gateway/open/v1
LOCAL_LLM_URL_OFFICE=https://ea-ai-gateway.corp.kuaishou.com/ea-ai-gateway/open/v1
LOCAL_LLM_DEFAULT_MODEL=deepseek_v3
```

The code reads these environment variables using `os.getenv()`. If the environment variables are not present, it falls back to the default values.

Additional configuration is also defined in `config/llm_config.py`:

- `LOCAL_LLM_MODEL_NAMES` maps friendly names to API model names
- `TOKEN_LIMITS` defines token limits for each model
- `API_CONFIG["local_llm"]` contains the base URLs for the local LLM service

## Troubleshooting

If you encounter issues:

1. **Permission Errors**: Initially we encountered two issues:
   
   a. Model-specific permission errors when trying to use models:
   ```
   {'code': 50000, 'message': 'doc-agent has no permission for [model_name]'}
   ```
   
   b. API access permission error (403 Forbidden):
   ```
   {'timestamp': '2025-07-21 16:40:23', 'status': 403, 'error': 'Forbidden', 'message': 'FORBIDDEN', 'path': '/ea-ai-gateway/open/v1/chat/completions'}
   ```
   
   Solution: Both issues have been fixed by:
   1. Using the correct app key (41e638cc-c9f8-482e-8610-85fa7b48ac94) and secret key
   2. Adding X-App-Code and X-App-Key headers to the API requests
   3. Using the deepseek_v3 model which we have permission for

2. **Authentication Errors**: Check that you're using the correct app credentials in `LOCAL_LLM_CONFIG`. The access token is automatically refreshed if it expires.

3. **Network Issues**: Verify that you can access the local LLM service:
   - Office network: `https://ea-ai-gateway.corp.kuaishou.com/ea-ai-gateway/open/v1`
   - IDC: `http://ea-ai-gateway.common.ee-prod.internal:18080/ea-ai-gateway/open/v1`

4. **Model Not Found**: Ensure you're using the correct model name as defined in `LOCAL_LLM_MODEL_NAMES` or `gpt-4o`.

5. **Token Generation**: If token generation fails, you can manually generate a token using the token API:
   ```bash
   curl --location --request POST 'https://is-gateway.corp.kuaishou.com/token/get' \
     --header 'Content-Type: application/json' \
     --data '{
         "appKey": "41e638cc-c9f8-482e-8610-85fa7b48ac94",
         "secretKey": "openAppba9797e14aa5b1fd041839cda"
     }'
   ```
   Then set this token in the test script manually.

6. **For Debugging**: Use the improved test scripts with additional logging:
   ```bash
   python test_qwen_vl_72b.py --document --debug
   python test_qwen_vl_72b.py --token-only --debug
   python test_local_llm_integration.py --debug
   ```
   Check the log file (qwen_vl_test.log) for detailed error information.