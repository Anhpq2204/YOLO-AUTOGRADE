import pandas as pd

def grade_he(dict_grade):
    mdt = str(dict_grade['MDT']).strip()

    # Đảm bảo MDT đọc dưới dạng chuỗi
    df = pd.read_excel('grade.xlsx', dtype={'MDT': str})

    df_grade = df[df['MDT'] == mdt]
    if df_grade.empty:
        raise ValueError(f"[!] Không tìm thấy MDT = {mdt} trong grade.xlsx")

    # Lấy duy nhất một hàng dữ liệu
    row = df_grade.iloc[0]

    # ----- Part 1 -----
    stu_grade_1 = 0
    for i, ans_stu in dict_grade['Part1'].items():
        col = f'P1.{i}'
        if col in row:
            stu_grade_1 += 0.25 * int(ans_stu == str(row[col]).strip())
        else:
            print(f"[!] Thiếu cột {col}")

    # ----- Part 2 -----
    stu_grade_2 = 0
    part2 = dict_grade['Part2']
    count = [0] * 5
    for i in range(1, 14, 4):
        correct = 0
        grp = int(i / 4 + 1)
        for j in range(4):
            key = i + j
            col = f'P2.{grp}.{j+1}'
            if col in row:
                if part2.get(key, '').lower() == str(row[col]).strip().lower():
                    correct += 1
        count[correct] += 1
    stu_grade_2 = count[1]*0.1 + count[2]*0.25 + count[3]*0.5 + count[4]*1

    # ----- Part 3 -----
    stu_grade_3 = 0
    part3 = dict_grade['Part3']
    for i, ans_stu in part3.items():
        col = f'P3.{i}'
        if col in row:
            stu_grade_3 += 0.5 * int(ans_stu == str(row[col]).strip())
        # else:
        #     print(f"[!] Thiếu cột {col}")

    return stu_grade_1 + stu_grade_2 + stu_grade_3
