# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
from datetime import datetime
import logging
import os
import platform
import subprocess


# 掘金终端需要打开，接口取数是通过网络请求的方式
# 设置token，可在用户-密钥管理里查看获取已有token ID


class GmDataService:
    """
    Service for download market data from GM
    """

    def __init__(self):
        self.SECOND_PATH = "C:\\Users\\Administrator\\Desktop\\secondary_daily_market_data\\"
        self.MAIN_PATH = "C:\\Users\\Administrator\\Desktop\\hot_daily_market_data\\"
        self.LOG_PATH = "C:\\Users\\Administrator\\Desktop\\hot_daily_log\\"
        self.OVER_PATH = "C:\\Users\\Administrator\\Desktop\\over_daily_market_data\\"

        # 配置日志记录
        self.setup_logging()

        # 加载配置
        token_id = "84d6e54e58a4989e1a6fb72b7a5ed217c0382df9"
        set_token(token_id)
        # set_token('84d6e54e58a4989e1a6fb72b7a5ed217c0382df9')

        # 配置错误日志记录并获取对应的日志记录器
        self.error_logger = self.setup_error_logging()

    def setup_logging(self):
        """
        设置日志记录的配置，每天生成一个日志文件，日志级别为INFO，格式包含时间、日志级别和消息内容，
        日志文件存放在当前目录下的logs文件夹中（若不存在则创建），文件名为当前日期.log
        """
        log_dir = self.LOG_PATH
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_date = datetime.now().strftime('%Y-%m-%d')
        log_file_path = os.path.join(log_dir, f"{current_date}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='a'),  # 'a'表示追加模式
                logging.StreamHandler()  # 同时在控制台输出日志，方便查看实时信息，可根据需求去掉
            ]
        )

    def setup_error_logging(self):
        """
        配置错误日志记录，设置将错误信息记录到指定的文件error.log中，
        同时设置日志格式，并返回配置好的logger对象。
        """
        # 创建logger对象
        logger = logging.getLogger('error_logger')
        logger.setLevel(logging.ERROR)

        # 创建文件处理器，用于将日志写入文件
        file_handler = logging.FileHandler('error.log')
        file_handler.setLevel(logging.ERROR)

        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到logger对象
        logger.addHandler(file_handler)

        return logger

    def get_all_hot(self, trade_date, actual=False):
        df = get_symbols(
            sec_type1=1040,
            trade_date=trade_date,
            df=True
        )
        contract_list = []
        if len(df) <= 0:
            # 使用日志记录代替print
            logging.info(f"当天没有合约数据！, trade_date: {trade_date}")
            return contract_list

        for sec_id, ex in zip(df.sec_id, df.exchange):
            contract_list.append(f"{ex}.{sec_id}")

        result = [''.join(filter(lambda x: not x.isdigit(), contract)).upper() for contract in contract_list]
        logging.info('-' * 50)
        logging.info(u'主力合约确认')
        result_abbr = set(result)
        if actual:
            result_actual_list = []
            for abbr_symbol in result_abbr:
                hot_symbol_list = fut_get_continuous_contracts(csymbol=abbr_symbol, start_date=trade_date,
                                                               end_date=trade_date)
                if hot_symbol_list:
                    result_actual_list.append(hot_symbol_list[-1]["symbol"])
                    # data = history_n(symbol=hot_symbol_list[-1]["symbol"], frequency='1800s', count=200, end_time=datetime.now(), fields=['close'], df=True)
                    # xx = np.array(data['close'])
            return result_actual_list
        else:
            return result_abbr

    def check_last_day(self, df_main, df_second):
        last_row_main = df_main.iloc[-1]
        last_row_second = df_second.iloc[-1]

        if last_row_main.name == last_row_second.name:
            main_symbol = last_row_main['symbol'].split('.')[0]
            second_symbol = last_row_second['symbol'].split('.')[0]
            main_symbol_num = int(''.join(filter(str.isdigit, main_symbol)))
            second_symbol_num = int(''.join(filter(str.isdigit, second_symbol)))
            if second_symbol_num > main_symbol_num and last_row_second['volume'] > last_row_main['volume'] and \
                    last_row_second['open_interest'] > last_row_main['open_interest']:
                df_main.iloc[-1] = last_row_second
                df_second.iloc[-1] = last_row_main

        return df_main, df_second

    def query_export_symbol(self, symbol, startdt, enddt, insert_mode=False):
        main_result_df = self.query_hot_history(symbol, startdt, enddt, secondary=False)
        second_result_df = self.query_hot_history(symbol, startdt, enddt, secondary=True)

        if main_result_df.empty or second_result_df.empty :
            pass
        else:
            main_result_df, second_result_df = self.check_last_day(main_result_df, second_result_df)
            symbol_name = symbol.split(".")[1]
            main_path = self.MAIN_PATH + symbol_name + ".csv"
            second_path = self.SECOND_PATH + symbol_name + ".csv"
            if insert_mode:
                self.insert_csv(main_result_df, main_path)
                self.insert_csv(second_result_df, second_path)
            else:
                self.save_csv_more(main_result_df, main_path)
                self.save_csv_more(second_result_df, second_path)
            logging.info(f'合约{symbol} done')

    def query_hot_history(self, symbol, start, end, secondary=False):
        """
        Query history bar data from GmData
        """
        if secondary:
            symbol = symbol + "22"
            logging.info(f'输出次主力合约{symbol}')
        else:
            logging.info(f'输出主力合约{symbol}')
        symbol_data = get_history_symbol(symbol=symbol, start_date=start, end_date=end, df=True)
        try:
            history_data = history(symbol=symbol, frequency='1d', start_time=start, end_time=end,
                                   adjust=ADJUST_PREV, df=True)
        except Exception as e:
            error_message = f"获取历史数据时出现错误: {e}"
            self.error_logger.error(error_message)
            self.open_error_file()
            history_data = pd.DataFrame()  # 如果报错，返回空的DataFrame
        if history_data.empty:
            logging.info(f"获取历史数据时出现错误: {symbol}")
            return history_data
        else:
            merged_df = pd.merge(history_data, symbol_data[['trade_date', 'upper_limit', 'lower_limit', 'settle_price']],
                                 left_on='eob', right_on='trade_date', how='left')
            hot_symbol_list = fut_get_continuous_contracts(csymbol=symbol, start_date=start, end_date=end)
            queue_df = pd.DataFrame(hot_symbol_list)
            merged_df['trade_date'] = merged_df['trade_date'].dt.strftime('%Y-%m-%d')
            merged_df.drop(columns=['symbol', 'frequency', 'bob', 'eob'], inplace=True)
            new_merged_df = pd.merge(merged_df, queue_df, on='trade_date', how='left')
            new_merged_df['trade_date'] = pd.to_datetime(new_merged_df['trade_date'])
            new_merged_df = new_merged_df.dropna()
            new_merged_df['symbol'] = new_merged_df['symbol'].apply(
                lambda symbol: f"{symbol.split('.')[1]}.{symbol.split('.')[0]}" if len(symbol.split('.')) == 2 else symbol)

            new_merged_df.set_index('trade_date', inplace=True)
            new_merged_df.index.name = None
            new_merged_df.rename(columns={'amount': 'money', 'position': 'open_interest', 'upper_limit': 'high_limit',
                                          'lower_limit': 'low_limit', 'settle_price': 'avg'}, inplace=True)
            new_merged_df = new_merged_df[
                ['symbol', 'open', 'high', 'low', 'close', 'volume', 'money', 'high_limit', 'low_limit', 'avg',
                 'open_interest', 'pre_close']]

            return new_merged_df

    def insert_csv(self, resultData, path=None):
        last_row_second = resultData.iloc[-1]
        last_row_second.name = datetime.strftime(last_row_second.name, '%Y-%m-%d')

        try:
            df_output = pd.read_csv(path)
            df_output = df_output.set_index(df_output.columns[0])
            df_output.index.name = None
        except FileNotFoundError:
            logging.warning(f"文件不存在错误{path}文件未找到。")
            return
        last_date_output = df_output.index[-1] if len(df_output) > 0 else None
        last_date_second = last_row_second.name

        if last_date_output == last_date_second:
            df_output.iloc[-1] = last_row_second
        else:
            df_output = pd.concat([df_output, pd.DataFrame([last_row_second])], ignore_index=False)

        with open(path, "w", encoding='utf-8-sig') as f:
            df_output.to_csv(path)
        logging.info(f"insert {path} complete")
    
    def replace_last_row_zeros(self, df, check_columns=None):
        """
        检查并替换最后一行指定列的零值
        
        参数：
        df : pd.DataFrame - 需要处理的数据框
        check_columns : list - 需要检查的列名列表（默认为常见股票字段）
        
        返回：
        pd.DataFrame - 处理后的数据框
        """
        # 设置默认检查列
        if check_columns is None:
            check_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 列存在性校验
        missing_cols = [col for col in check_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺失必要列：{missing_cols}")

        # 至少需要两行数据才能执行替换
        if len(df) < 2:
            return df

        # 获取最后两行索引
        last_idx = df.index[-1]
        prev_idx = df.index[-2]

        # 检查最后一行是否存在零值
        last_row = df.loc[last_idx, check_columns]
        if (last_row == 0).any():
            # 使用上一行对应列的值替换
            df.loc[last_idx, check_columns] = df.loc[prev_idx, check_columns].values
            logging.info("!!!!!!!!!!!已执行最后一行的零值替换!!!!!!!!!!!!!!!!!")
        else:
            logging.info("最后一行无零值，无需替换")

        return df

    def save_csv_more(self, resultData, path=None) -> None:
        """
        Save table data into a csv file
        """
        if not path:
            return
        if pd.api.types.is_datetime64_any_dtype(resultData.index):
            resultData.index = resultData.index.strftime('%Y-%m-%d')
        with open(path, "w", encoding='utf-8-sig') as f:
            resultData = self.replace_last_row_zeros(resultData)
            resultData.to_csv(path)
        logging.info(f"renew {path} complete")

    def export_all_symbols_info(self, trade_date=None, out_path=None, sec_type1=1040):
        """
        导出指定交易日所有合约信息到 CSV（默认 trade_date 为当天）。
        使用 get_symbols 获取合约列表，使用 get_instrumentinfos 获取每个合约详细信息。

        参数:
            trade_date: str (YYYY-MM-DD) or None 默认为当天
            out_path: 输出文件路径，默认放在 MAIN_PATH/all_instruments_info.csv
            sec_type1: int, 交易品种类型，默认 1040
        返回:
            out_path 字符串或 None
        """
        if trade_date is None:
            trade_date = datetime.strftime(datetime.today(), '%Y-%m-%d')

        if not out_path:
            out_path = os.path.join(self.MAIN_PATH, "all_instruments_info.csv")

        try:
            symbols_df = get_symbols(sec_type1=sec_type1, trade_date=trade_date, df=True)
        except Exception as e:
            logging.error(f"获取合约列表失败: {e}")
            return None

        if symbols_df is None or len(symbols_df) == 0:
            logging.info(f"{trade_date} 没有发现合约数据")
            return None

        records = []
        for _, row in symbols_df.iterrows():
            sec_id = row.get('sec_id') if 'sec_id' in row else None
            exchange = row.get('exchange') if 'exchange' in row else None
            symbol_full = f"{exchange}.{sec_id}" if (exchange and sec_id) else None

            info_record = {}
            try:
                info_record.update(row.to_dict())
            except Exception:
                pass

            # 尝试获取更详细的合约信息
            try:
                inst_info = None
                try:
                    inst_info = get_instrumentinfos(symbol=symbol_full, df=True)
                except Exception:
                    try:
                        inst_info = get_instrumentinfos(sec_id=sec_id, df=True)
                    except Exception:
                        inst_info = None

                if inst_info is not None:
                    if hasattr(inst_info, "to_dict"):
                        recs = inst_info.to_dict(orient='records')
                        if recs:
                            info_record.update(recs[0])
                    elif isinstance(inst_info, dict):
                        info_record.update(inst_info)
            except Exception as e:
                logging.warning(f"获取 {symbol_full} 详细信息失败: {e}")

            if symbol_full:
                info_record['symbol_full'] = symbol_full

            records.append(info_record)

        if not records:
            logging.info("无可导出的合约信息")
            return None

        df_out = pd.DataFrame.from_records(records)
        # 把 symbol_full 放在首列（若存在）
        cols = df_out.columns.tolist()
        if 'symbol_full' in cols:
            cols.insert(0, cols.pop(cols.index('symbol_full')))
            df_out = df_out[cols]

        # 过滤：严格检查 symbol_full（去掉空白/引号），只保留不包含任何数字的行
        # 并在日志中输出被过滤掉的示例（前 N 行）以便复查
        if 'symbol_full' in df_out.columns:
            before = len(df_out)
            # 规范化字符串并检查数字
            sf = df_out['symbol_full'].astype(str).str.replace('"', '').str.strip()
            mask_digits = sf.str.contains(r"\d", na=False)
            keep_mask = ~mask_digits
            removed = before - keep_mask.sum()
            if removed > 0:
                logging.info(f"过滤掉包含数字的symbol_full行: {removed} 行（保留不含数字的行）")
                # 输出被过滤的前 N 个示例，方便人工复查
                N = 10
                examples = df_out.loc[mask_digits, 'symbol_full'].astype(str).head(N).tolist()
                logging.info(f"被过滤的示例 (最多 {N}): {examples}")

            # 只保留不含数字的行
            df_out = df_out[keep_mask]

            # 去重：如果有相同的 symbol_full，只保留第一条
            before2 = len(df_out)
            df_out = df_out.drop_duplicates(subset='symbol_full', keep='first')
            dup_removed = before2 - len(df_out)
            if dup_removed > 0:
                logging.info(f"去重重复symbol_full: 删除 {dup_removed} 行")

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
            logging.info(f"已导出合约信息到 {out_path}")
            return out_path
        except Exception as e:
            logging.error(f"写入文件失败 {out_path}: {e}")
            return None

    def open_error_file(self):
        """
        根据不同的操作系统，使用相应的命令来打开指定路径的文件（这里是error.log）。
        在Windows上使用start命令，在Linux和macOS上使用open命令（适用于大多数情况）。
        """
        system = platform.system()
        if system == "Windows":
            os.startfile('error.log')
        elif system == "Linux":
            subprocess.call(['open', 'error.log'])
        elif system == "Darwin":
            subprocess.call(['open', 'error.log'])
    
    def process_main_contract_rollover(self):
        """
        处理主力合约切换，将切换后下一个交易日的旧合约数据保存到OVER_PATH
        
        功能：
        1. 读取MAIN_PATH下的主力合约历史数据
        2. 检测合约切换事件
        3. 下载切换后下一个交易日的旧合约数据
        4. 保存到OVER_PATH目录
        """
        logging.info("开始处理主力合约切换数据...")
        
        # 确保输出目录存在
        os.makedirs(self.OVER_PATH, exist_ok=True)
        
        # 获取MAIN_PATH下的所有CSV文件
        main_files = [f for f in os.listdir(self.MAIN_PATH) if f.endswith('.csv')]
        
        for file in main_files:
            base_symbol = file.split('.')[0].upper()
            main_file_path = os.path.join(self.MAIN_PATH, file)
            
            try:
                # 读取主力合约数据
                df_main = pd.read_csv(main_file_path, index_col=0, parse_dates=True)
                logging.info(f"成功读取 {base_symbol} 的主力合约数据")
                
                # 检测合约切换事件
                rollover_events = self.detect_contract_rollover(df_main, base_symbol)
                
                if not rollover_events:
                    logging.info(f"{base_symbol} 未检测到主力合约切换")
                    continue
                
                # 处理每个切换事件
                for event in rollover_events:
                    self.handle_rollover_event(event, base_symbol)
                    
            except Exception as e:
                logging.error(f"处理 {base_symbol} 时出现错误: {e}")
                self.error_logger.error(f"处理 {base_symbol} 时出现错误: {e}")
                continue
        
        logging.info("主力合约切换数据处理完成！")
    
    def detect_contract_rollover(self, df, base_symbol):
        """
        检测主力合约切换事件
        
        参数：
        df: DataFrame - 主力合约数据
        base_symbol: str - 基础品种代码
        
        返回：
        list - 切换事件列表，每个事件包含日期、旧合约和新合约
        """
        rollover_events = []
        
        # 按日期排序
        df = df.sort_index()
        
        # 记录前一个合约
        prev_contract = df['symbol'].iloc[0]
        
        for i in range(1, len(df)):
            curr_contract = df['symbol'].iloc[i]
            curr_date = df.index[i]
            
            # 检测合约切换
            if curr_contract != prev_contract:
                # 记录切换事件
                rollover_events.append({
                    'date': curr_date,
                    'prev_contract': prev_contract,
                    'curr_contract': curr_contract
                })
                logging.info(f"检测到 {base_symbol} 合约切换: {prev_contract} -> {curr_contract} (日期: {curr_date})")
                prev_contract = curr_contract
        
        return rollover_events

        
    def handle_rollover_event(self, event, base_symbol):
        """
        处理单个合约切换事件，下载并保存旧合约下一个交易日的数据
        
        参数：
        event: dict - 切换事件，包含日期、旧合约和新合约
        base_symbol: str - 基础品种代码
        """
        rollover_date = event['date']
        prev_contract = event['prev_contract']
        
        # 计算下一个交易日
        next_trading_date = rollover_date + pd.Timedelta(days=2)
        
        # 格式化日期字符串
        start_date = rollover_date.strftime('%Y-%m-%d')
        end_date = next_trading_date.strftime('%Y-%m-%d')
        
        try:
            # 使用history函数下载数据
            logging.info(f"正在下载 {prev_contract} 在 {start_date} 的数据")
            symbol, exchange = prev_contract.split(".")
            gm_symbol = f"{exchange}.{symbol}"
            
            # 调用history函数获取数据
            history_data = history(
                symbol=gm_symbol,
                frequency='1d',
                start_time=start_date,
                end_time=end_date,
                adjust=ADJUST_PREV,
                df=True
            )
            
            if history_data.empty:
                logging.warning(f"未获取到 {prev_contract}, {gm_symbol}在 {start_date} 的数据")
                return
            
            # 处理数据
            processed_data = self.process_history_data(history_data, prev_contract)
            
            if processed_data.empty:
                logging.warning(f"处理后 {prev_contract} 在 {start_date} 的数据为空")
                return
            
            # 保存数据到OVER_PATH
            self.save_rollover_data(processed_data, base_symbol)
            
        except Exception as e:
            logging.error(f"下载 {prev_contract} 在 {start_date} 的数据时出错: {e}")
            self.error_logger.error(f"下载 {prev_contract} 在 {start_date} 的数据时出错: {e}")
    
    def process_history_data(self, history_data, symbol):
        """
        处理下载的历史数据，使其格式与主力合约数据一致
        
        参数：
        history_data: DataFrame - 下载的历史数据
        symbol: str - 合约代码
        
        返回：
        DataFrame - 处理后的数据
        """
        try:
            # 重命名列名以匹配主力合约数据格式
            column_mapping = {
                'amount': 'money',
                'position': 'open_interest',
                'eob': 'trade_date'
            }
            
            history_data = history_data.rename(columns=column_mapping)
            
            # 设置日期索引，确保格式为YYYY-MM-DD
            if 'trade_date' in history_data.columns:
                history_data['trade_date'] = pd.to_datetime(history_data['trade_date'])
                # 转换为YYYY-MM-DD格式的字符串
                history_data['trade_date'] = history_data['trade_date'].dt.strftime('%Y-%m-%d')
                # 再转换回datetime类型并设置为索引
                history_data['trade_date'] = pd.to_datetime(history_data['trade_date'])
                history_data.set_index('trade_date', inplace=True)
            
            # 使用传入的symbol替代gm返回的特殊格式symbol
            history_data['symbol'] = symbol
            
            # 只保留需要的列
            required_columns = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'money', 'open_interest']
            history_data = history_data[required_columns]
            
            # 处理缺失值
            history_data = history_data.dropna()
            
            return history_data
            
        except Exception as e:
            logging.error(f"处理历史数据时出错: {e}")
            self.error_logger.error(f"处理历史数据时出错: {e}")
            return pd.DataFrame()
    
    def save_rollover_data(self, data, base_symbol):
        """
        保存主力合约切换数据到OVER_PATH
        
        参数：
        data: DataFrame - 要保存的数据
        base_symbol: str - 基础品种代码
        """
        over_file_path = os.path.join(self.OVER_PATH, f'{base_symbol}.csv')
        
        try:
            # 检查文件是否存在
            if os.path.exists(over_file_path):
                # 文件存在，读取现有数据
                existing_df = pd.read_csv(over_file_path, index_col=0, parse_dates=True)
                # 合并数据
                combined_df = pd.concat([existing_df, data])
                # 去重，按日期保留最新数据
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                # 保存数据
                self.save_csv_more(combined_df, over_file_path)
                logging.info(f"数据已追加到 {over_file_path}")
            else:
                # 文件不存在，直接保存
                self.save_csv_more(data, over_file_path)
                logging.info(f"数据已保存到 {over_file_path}")
                
        except Exception as e:
            logging.error(f"保存 {base_symbol} 的切换数据时出错: {e}")
            self.error_logger.error(f"保存 {base_symbol} 的切换数据时出错: {e}")


if __name__ == '__main__':
    startdt = '2020-01-15'
    enddt = datetime.strftime(datetime.today(), '%Y-%m-%d')
    # enddt = '2024-09-19'
    gmService = GmDataService()
    # gmService.export_all_symbols_info()
    
    # # 导出所有主力合约数据
    # hot_future_list = gmService.get_all_hot(enddt)
    # for symbol in hot_future_list:
        # logging.info('-' * 50)
        # gmService.query_export_symbol(symbol, startdt, enddt)
    
    # 处理主力合约切换数据
    gmService.process_main_contract_rollover()
