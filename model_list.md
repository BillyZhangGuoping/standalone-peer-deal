# 趋势信号模型和资金分配模型清单

## 一、趋势信号模型

### 1. TrendSignalFusionModel
- **实现文件**：`trend_models.py:23`
- **功能**：趋势信号融合模型，封装了`trend_signal_fusion.py`的功能
- **输入**：历史数据DataFrame、指定日期、品种列表
- **输出**：包含品种、趋势方向和信号强度的DataFrame

### 2. RandomForestModel
- **实现文件**：`trend_models.py:98`
- **功能**：随机森林模型，封装了`random_forest_signal_generation.py`的功能
- **输入**：品种数据字典、指定日期、品种列表
- **输出**：包含品种、趋势方向和信号强度的DataFrame

## 二、资金分配模型

### 1. RiskParityAllocation
- **实现文件**：`allocation_methods.py:18`
- **功能**：风险平价资金分配方法，封装了`risk_parity_allocation.py`的功能
- **输入**：总资金、品种数据字典、指定日期、历史数据
- **输出**：资金分配字典和风险单位字典

### 2. calculate_atr_allocation
- **实现文件**：`risk_allocation.py:7`
- **功能**：基于ATR的等风险资金分配
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）
- **输出**：资金分配字典和风险单位字典

### 3. floor_asset_tilt_allocation
- **实现文件**：`risk_allocation.py:81`
- **功能**：地板资产倾斜 (sign/Vol) 分配策略，倾向于给低波动品种分配更多头寸
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）、波动率计算窗口（默认20）
- **输出**：资金分配字典和风险单位字典

### 4. adaptive_atr_allocation
- **实现文件**：`risk_allocation.py:217`
- **功能**：自适应ATR窗口的分配策略，根据市场波动率状态调整ATR计算窗口
- **输入**：总资金、品种数据字典、波动率状态阈值（默认0.3）
- **输出**：包含ATR信息的字典

### 5. atr_momentum_composite_allocation
- **实现文件**：`risk_allocation.py:257`
- **功能**：ATR动量复合分配策略，结合ATR风险分配和动量加权
- **输入**：总资金、品种数据字典、动量计算窗口（默认20）
- **输出**：资金分配字典

### 6. enhanced_atr_allocation
- **实现文件**：`risk_allocation.py:340`
- **功能**：增强型ATR分配策略，综合考虑合约乘数、当前价格、趋势强度和保证金比率
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）
- **输出**：资金分配字典和风险单位字典

### 7. enhanced_atr_cluster_risk_allocation
- **实现文件**：`risk_allocation.py:432`
- **功能**：增强型ATR聚类风险分配策略，基于ATR分配并按聚类控制风险
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）
- **输出**：资金分配字典、风险单位字典和聚类权重字典

### 8. cluster_risk_parity_allocation
- **实现文件**：`risk_allocation.py:587`
- **功能**：基于相关性聚类和风险平价的分配策略，结合聚类关系、相关性和ATR
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）
- **输出**：资金分配字典、风险单位字典和聚类权重字典

### 9. enhanced_sharpe_atr_allocation
- **实现文件**：`risk_allocation.py:804`
- **功能**：基于夏普比率优化的增强型ATR分配策略，最大化夏普比率和收益率
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）、市场参数
- **输出**：资金分配字典和风险单位字典

### 10. signal_strength_based_allocation
- **实现文件**：`risk_allocation.py:1051`
- **功能**：基于信号强度的风险分配，使用价格序列回归斜率的t统计量作为信号强度
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）、波动率计算窗口（默认20）
- **输出**：资金分配字典和风险单位字典

### 11. model_based_allocation
- **实现文件**：`risk_allocation.py:1182`
- **功能**：基于LightGBM模型的智能分配策略，每80天重新训练模型
- **输入**：总资金、品种数据字典、目标波动率（默认0.01）、市场参数
- **输出**：资金分配字典和风险单位字典

## 三、使用说明

1. **趋势信号模型**：通过配置`config.json`中的`trend_model.type`字段选择，可选值为：
   - `trend_signal_fusion`：使用TrendSignalFusionModel
   - `random_forest`：使用RandomForestModel

2. **资金分配模型**：通过配置`config.json`中的`allocation_method.type`字段选择，可选值为：
   - `risk_parity`：使用RiskParityAllocation
   - `calculate_atr_allocation`：使用calculate_atr_allocation
   - `floor_asset_tilt_allocation`：使用floor_asset_tilt_allocation
   - 其他资金分配函数也可直接在代码中调用

3. **配置示例**：
   ```json
   {
     "trend_model": {
       "type": "random_forest",
       "params": {}
     },
     "allocation_method": {
       "type": "calculate_atr_allocation",
       "params": {}
     }
   }
   ```

这些模型提供了多样化的趋势信号生成和资金分配方案，可以根据不同的市场环境和策略需求进行选择和配置。