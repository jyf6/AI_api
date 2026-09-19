from providers.openai.backend import OpenAIBackendAPI


def test_sse_image_quota_message_parses_chinese_reset_duration():
    event = {
        "v": {
            "message": {
                "author": {"role": "assistant"},
                "content": {"parts": ["你已达到 Free 套餐的图像生成请求上限。上限将在 7 小时后重置。"]},
            },
        },
    }

    assert OpenAIBackendAPI._sse_image_quota_error(event) == (
        "你已达到 Free 套餐的图像生成请求上限。上限将在 7 小时后重置。", 25230,
    )


def test_sse_image_quota_message_ignores_user_prompt():
    event = {
        "v": {
            "message": {
                "author": {"role": "user"},
                "content": {"parts": ["请生成图片，额度上限将在 7 小时后重置"]},
            },
        },
    }

    assert OpenAIBackendAPI._sse_image_quota_error(event) is None


def test_generated_image_ids_exclude_uploaded_reference_ids():
    uploaded_id = "file_00000000aaaaaaaaaaaaaaaaaaaaaaaa"
    generated_id = "file_00000000bbbbbbbbbbbbbbbbbbbbbbbb"
    mapping = {
        "user": {"message": {"author": {"role": "user"}, "content": {"parts": [f"file-service://{uploaded_id}"]}}},
        "image-tool": {"message": {"author": {"role": "tool"}, "metadata": {"async_task_type": "image_gen"}, "content": {"parts": [f"file-service://{uploaded_id}", f"file-service://{generated_id}"]}}},
    }

    assert OpenAIBackendAPI._generated_image_file_ids(mapping, {uploaded_id}) == [generated_id]


def test_generated_image_ids_include_image_generation_tool_output():
    uploaded_id = "file_00000000aaaaaaaaaaaaaaaaaaaaaaaa"
    generated_id = "file_00000000bbbbbbbbbbbbbbbbbbbbbbbb"
    mapping = {
        "user": {"message": {"author": {"role": "user"}, "content": {"parts": [f"file-service://{uploaded_id}"]}}},
        "image-tool": {
            "message": {
                "author": {"role": "tool"},
                "metadata": {"async_task_type": "image_gen"},
                "content": {"parts": [f"file-service://{uploaded_id}", f"file-service://{generated_id}"]},
            }
        },
    }

    assert OpenAIBackendAPI._generated_image_file_ids(mapping, {uploaded_id}) == [generated_id]


def test_generated_image_asset_ids_keep_sediment_output_separate():
    mapping = {
        "image-tool": {
            "message": {
                "author": {"role": "tool"},
                "metadata": {"async_task_type": "image_gen"},
                "content": {"parts": ["sediment://generated-attachment"]},
            }
        },
    }

    assert OpenAIBackendAPI._generated_image_asset_ids(mapping, set()) == ([], ["generated-attachment"])


def test_sse_patch_event_extracts_generated_attachment_immediately():
    """GPT 图片成功的 Patch 事件将工具消息放入 v.message，必须直接提取。"""
    event = {
        "v": {
            "conversation_id": "conversation-id",
            "message": {
                "author": {"role": "tool"},
                "metadata": {"async_task_type": "image_gen"},
                "content": {"parts": [{"asset_pointer": "sediment://generated-attachment"}]},
            },
        },
    }

    assert OpenAIBackendAPI._sse_image_asset_ids(event, set()) == (
        "conversation-id", [], ["generated-attachment"],
    )


def test_sse_batch_patch_extracts_image_tool_message_with_inherited_conversation_id():
    """批量 Patch 内的工具消息也必须继承外层会话 ID 并成为生成结果。"""
    event = {
        "conversation_id": "conversation-id",
        "p": "",
        "o": "patch",
        "v": [{
            "p": "",
            "o": "add",
            "v": {
                "message": {
                    "id": "tool-message-id",
                    "author": {"role": "tool"},
                    "metadata": {"async_task_type": "image_gen"},
                    "content": {"parts": [{"asset_pointer": "file-service://file_00000000cccccccccccccccccccccccc"}]},
                },
            },
        }],
    }

    assert OpenAIBackendAPI._sse_image_asset_ids(event, set()) == (
        "conversation-id", ["file_00000000cccccccccccccccccccccccc"], [],
    )


def test_conversation_fallback_accepts_non_user_asset_when_image_tool_metadata_is_absent():
    """会话落盘后若缺少 image_gen 元数据，仍要取得非用户消息中的生成资源。"""
    generated_id = "file_00000000dddddddddddddddddddddddd"
    mapping = {
        "input": {"message": {"author": {"role": "user"}, "content": {"parts": [{"asset_pointer": "file-service://file_00000000aaaaaaaaaaaaaaaaaaaaaaaa"}]}}},
        "result": {"message": {"author": {"role": "assistant"}, "content": {"parts": [{"asset_pointer": f"file-service://{generated_id}"}]}}},
    }

    assert OpenAIBackendAPI._conversation_generated_image_asset_ids(mapping, set()) == (
        [generated_id], [],
    )


def test_sediment_file_identifier_is_not_misclassified_as_file_service_result():
    """sediment 附件的 file 标识只能走会话附件下载接口。"""
    asset_id = "file_00000000eeeeeeeeeeeeeeeeeeeeeeee"

    assert OpenAIBackendAPI._asset_ids_from_payload(
        {"asset_pointer": f"sediment://{asset_id}"}, set(),
    ) == ([], [asset_id])
