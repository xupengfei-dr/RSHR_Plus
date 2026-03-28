import torch



def restrict_logits_to_valid_range(logits, question_labels, label_to_valid_indices_map, num_total_answers):
    """
    根据每个问题类型，将logits限制在有效的答案下标范围内。

    参数:
        logits (torch.Tensor): 模型的原始输出logits，形状 (batch_size, num_total_answers)。
        question_labels (list or np.array or torch.Tensor): 每个问题的类型标签，长度 batch_size。
        label_to_valid_indices_map (dict): 问题类型到有效答案下标列表的映射。
        num_total_answers (int): 总的答案类别数量。

    返回:
        torch.Tensor: 修改后的logits，无效答案的logits被设置为极小值。
    """
    batch_size = logits.shape[0]
    modified_logits = logits.clone() # 创建副本以避免修改原始logits

    for i in range(batch_size):
        current_label = question_labels[i]
        if current_label in label_to_valid_indices_map:
            valid_indices = label_to_valid_indices_map[current_label]

            # 创建一个mask，有效答案位置为True (或0)，无效答案位置为False (或-infinity)
            # 方法1: 使用布尔掩码和masked_fill_
            mask = torch.zeros(num_total_answers, dtype=torch.bool, device=logits.device)
            if valid_indices: # 确保valid_indices不为空
                mask[torch.tensor(valid_indices, device=logits.device)] = True # 有效位置设为True

            # 对于不在有效范围内的答案，将其logit设置为一个非常小的值（接近负无穷）
            # 这样在softmax之后，它们的概率会趋近于0
            modified_logits[i].masked_fill_(~mask, float('-inf'))

            # 方法2: 创建一个加性掩码 (更直接，但要注意浮点精度)
            # additive_mask = torch.full((num_total_answers,), float('-inf'), device=logits.device)
            # if valid_indices:
            #     additive_mask[torch.tensor(valid_indices, device=logits.device)] = 0.0
            # modified_logits[i] = logits[i] + additive_mask
        else:
            # 如果问题类型不在映射中，可以选择：
            # 1. 不做任何修改 (当前实现)
            # 2. 记录一个警告
            # 3. 将所有logit设为-inf (强制不预测或预测一个“未知类型”类别，如果存在)
            print(f"警告: 问题类型 '{current_label}' 在样本 {i} 中未找到有效的答案范围映射。未进行限制。")

    return modified_logits

# --- 示例用法 ---
if __name__ == '__main__':
    # 假设的参数
    batch_size = 4
    num_total_answers = 15 # 总共有15个可能的答案类别
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟分类头的输出logits
    mock_logits = torch.randn(batch_size, num_total_answers, device=device)
    print("原始Logits:\n", mock_logits)

    # 模拟问题类型标签
    mock_question_labels = ['count', 'yes_no', 'count', 'unknown_type'] # 注意有一个未知类型

    # 定义问题类型到有效答案范围的映射
    label_to_valid_indices_map_example = {
        'count': list(range(3, 9)),       # 答案类别 3, 4, 5, 6, 7, 8
        'yes_no': [0, 1,2],                 # 答案类别 0, 1
        'color': list(range(9, 13)),      # 答案类别 9, 10, 11, 12
    }

    # 应用限制
    restricted_logits = restrict_logits_to_valid_range(
        mock_logits,
        mock_question_labels,
        label_to_valid_indices_map_example,
        num_total_answers
    )
    print("\n限制后的Logits:\n", restricted_logits)
    print("\n限制后的Logits:\n", restricted_logits.shape)

    # 验证：对限制后的logits应用softmax，看看非有效范围的概率是否接近0
    restricted_probs = torch.softmax(restricted_logits, dim=-1)
    print("\n限制后的概率:\n", restricted_probs.shape)

    # 获取最终预测
    predicted_indices = torch.argmax(restricted_logits, dim=-1)
    print("\n最终预测的答案索引:\n", predicted_indices)
    print("\n最终预测的答案索引:\n", predicted_indices.shape)

    # 检查预测是否在有效范围内
    for i in range(batch_size):
        label = mock_question_labels[i]
        pred_idx = predicted_indices[i].item()
        if label in label_to_valid_indices_map_example:
            valid_range = label_to_valid_indices_map_example[label]
            is_valid = pred_idx in valid_range
            print(f"样本 {i} (类型: {label}): 预测索引 {pred_idx}, 有效范围 {valid_range}, 是否在有效范围内: {is_valid}")
            if not is_valid and valid_range : # 如果不在有效范围且有效范围不为空
                 # 理论上，如果正确实现了masked_fill_为-inf，argmax不应该选到无效范围
                 # 除非所有有效范围的logits也都是-inf（例如，valid_indices为空或者模型对有效答案完全没信心）
                 # 或者valid_indices为空列表
                 print(f"  注意: 样本 {i} 的预测超出了定义的有效范围！这不应该发生，除非所有有效logit也极低。")
        else:
            print(f"样本 {i} (类型: {label}): 无有效范围定义，预测索引 {pred_idx}")