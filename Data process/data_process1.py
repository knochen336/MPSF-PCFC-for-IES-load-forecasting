import csv

# 读取PSV文件并写入CSV文件
def convert_psv_to_csv(psv_file, csv_file):
    with open(psv_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter='|')
        with open(csv_file, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(row)

# 示例用法
psv_file = 'data/GHCNh_USW00023183_2023.psv'  # PSV文件路径
csv_file = 'data_pro/GHCNh_USW00023183_2023.csv'  # CSV文件输出路径
convert_psv_to_csv(psv_file, csv_file)
