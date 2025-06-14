import pandas as pd


def grade(dict_grade):
    sbd=dict_grade['SBD']
    mdt=dict_grade['MDT']
    df=pd.read_csv('grade.csv')
    df_grade=df[df['MDT']==mdt]
    # Cham part 1
    stu_grade_1=0
    part_1=dict_grade['Part1']
    for i in part_1.key():
        stu_grade_1=stu_grade_1+0.25*int(part_1[i]==df[f'1.{i}'])

    
    stu_grade_2=0
    part_2=dict_grade['Part2']
    count=[0]*5
    for i in range(1,14,4):
        cnt=0
        for j in range(4):
            if part_2[i+j]==df[f'2.{i}.{j+1}']:
                cnt=cnt+1
        count[cnt]=count[cnt]+1
    stu_grade_2=count[1]*0.1+count[2]*0.25+count[3]*0.5+count[4]*1

    stu_grade_3=0
    part_3=dict_grade['Part3']
    for i in part_2.key():
        stu_grade_2=stu_grade_3+0.5*int(part_3[i]==df[f'3.{i}'])

    
        



