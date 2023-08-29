import base64
import uuid
import json
import rich.json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image
import altair as alt
from pygments.lexers import go
from streamlit_lottie import st_lottie
import requests
import pdfkit
import os


def find_indices_and_values_col(data_frame, value1, value2):
    aa1 = data_frame.iloc[0, :].values
    indices1 = np.where(aa1 == value1)[0]
    aa12 = indices1[0]

    indices11 = np.where(aa1 == value2)[0]
    aa13 = indices11[0]

    return aa12, aa13



def calculate_arr(matrix):
    b = len(matrix)
    matb = np.identity(b)  # 生成单位矩阵
    np_matrix1 = matb - matrix
    a = np.linalg.inv(np_matrix1)  # 对矩阵取逆
    return a

def download_link(df, file_name, text):
    csv = df.to_csv(index=True).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{file_name}">{text}</a>'

def download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href


def calculate_arr1(matrix):
    b = len(matrix)
    matb = np.identity(b)  # 生成单位矩阵
    np_matrix1 = matb - matrix
    return np_matrix1


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def turn(df):
    rows = slice(0, None)  # 假设需要选择第2行到最后一行
    cols = slice(0, None)  # 假设需要选择第2列到最后一列
    # 从DataFrame对象中选择指定范围的数字，并转换为矩阵
    matrix = df.iloc[rows, cols].values.astype(float)
    return matrix


def caler(file_path, file1):
    sheet_names = list(pd.read_excel(file_path, sheet_name=None).keys())
    sheet_names1 = list(pd.read_excel(file1, sheet_name=None).keys())

    # 使用 Pandas 读取文件内容
    # 读取选定工作表的内容
    ws3 = pd.read_excel(file1, sheet_name=sheet_names1[0], index_col=0)

    wm = pd.read_excel(file_path, sheet_name=sheet_names[0], index_col=0)
    # 比对元素是否正确
    # local of row
    aa1 = wm.iloc[0, :].values

    value_to_find = "中间使用合计"
    indices1 = np.where(aa1 == value_to_find)[0]
    aa12 = indices1[0]
    value_to_find2 = "农产品"
    indices11 = np.where(aa1 == value_to_find2)[0]
    aa13 = indices11[0]

    # local of col
    aa2 = wm.iloc[:, 0].values
    value_to_find1 = "中间投入合计"
    indices2 = np.where(aa2 == value_to_find1)[0]
    aa22 = indices2[0]
    aa2 = aa2[0:aa22]
    value_to_find2 = "农产品"
    indices21 = np.where(aa2 == value_to_find2)[0]
    aa23 = indices21[0]
    aa2 = aa2[aa23:aa22]
    wdd = wm.iloc[aa13:aa12, aa23:aa22].values
    ws = pd.DataFrame(wdd, index=aa2, columns=aa2)
    ws1 = pd.DataFrame(wdd, index=aa2, columns=aa2)

    index_array = ws.index.values
    index_array1 = ws3.index.values
    set_a = set(index_array)
    set_b = set(index_array1)
    is_b_in_a = set_b.issubset(set_a)
    difference = list(set(set_b) - set(set_a))

    if is_b_in_a is False:
        st.warning("当前所上传文档中行业名称与投入产出表中行业命名不一致，请重新确认！")
        difference = list(set(set_b) - set(set_a))
        st.write("此为不一致行业:", difference)

    else:

        cc = ws3.shape[0]  # 获取相关子行业个数
        cd = ws.shape[0]  # 获取初始流量表元素个数

        # 生成加入虚拟子行业的新流量表
        prefix = '虚拟纯子行业'
        for i in range(cc):
            c = ws3.index[i]
            col_name = f'{prefix}_{c}'
            ws1 = ws1.assign(
                **{col_name: ws1.loc[:, c] * ws3.loc[c, :].values})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 ws3.iloc[i, 1]})
            bb = 1 - ws3.loc[c, :].values
            ws1.loc[:, c] = ws1.loc[:, c] * bb  # 更新列的值
            ws1.loc[col_name] = ws1.loc[c, :] * ws3.loc[c, :].values  # 新增行

            ws1.loc[c, :] = ws1.loc[c, :] * bb

        # 获取总产量
        c1, c2 = find_indices_and_values_col(wm, "最终使用合计", "最终使用合计")

        wdd2 = wm.iloc[aa13:aa12, c2 + 2].values
        m1 = ["总产出"]
        wd = pd.DataFrame(wdd2, index=aa2, columns=m1)

        wd1 = pd.DataFrame(wdd2, index=aa2, columns=m1)

        # 生成加入虚拟子行业后的总产量表
        for i in range(cc):
            c = ws3.index[i]
            col_name = f'{prefix}_{c}'
            # ws1 = ws1.assign(**{col_name: ws.loc[c, :] * ws3.iloc[i, 1]})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 float_arr[i]
            bb = 1 - ws3.loc[c, :].values
            wd.loc[col_name] = wd.loc[c, :] * ws3.loc[c, :].values
            # 新增行

            wd.loc[c, :] = wd.loc[c, :] * bb

        # 计算累积总和
        sw = turn(ws1)
        row_sums = np.sum(sw, axis=0)

        # 提取索引值
        index_values = ws1.index.values
        index_values2 = wd1.index.values
        # 选择所需的数字范围
        rows = slice(0, None)  # 假设需要选择第2行到最后一行
        cols = slice(0, None)  # 假设需要选择第2列到最后一列

        # 从DataFrame对象中选择指定范围的数字，并转换为矩阵
        matrix = ws1.iloc[rows, cols].values.astype(float)

        # 确定数据范围
        rows = slice(0, None)  # 假设需要选择第2行到最后一行
        cols = slice(0, None)  # 假设需要选择第2列到最后一列
        vb = wd.iloc[rows, cols].values.astype(float)
        ab = vb.flatten()
        b = np.reciprocal(ab)  # 对数组中所有元素求倒数

        # array = matrix.tolist()
        # mat1 = b * array  # 直接消耗系数
        b = np.diag(b)

        mat1 = np.dot(matrix, b)

        a = calculate_arr(mat1)  # 里昂惕夫逆矩阵

        d = calculate_arr1(mat1)  # I-A矩阵

        e = np.dot(d, vb)

        # 计算渗透值
        # 查找位置
        dff = pd.DataFrame(a, index=index_values, columns=index_values)
        # 计算累积总和
        total_df = 0
        b2 = cd - 1
        for i in range(cc):
            c = ws3.index[i]
            a = ws3.loc[c, :].values
            df = dff.iloc[0:cd, cd + i] * wd.iloc[cd + i, 0] / dff.iloc[cd + i, cd + i]
            total_df += df

        # 遍历列表中的每个 DataFrame，并将它们对应位置的值相加到 result 中
        result = total_df

        arr3 = wd.values
        my_array = np.array(result)  # 生成列向量
        arr4 = wd.iloc[0:cd, :].values  # 这是加权处理过得到数值

        my_array1 = np.array(arr4).T  # 生成列向量
        # my_array44 = wd1.iloc[0:cd, :].values
        # my_array45 = np.array(my_array44).T  # 生成列向量
        my_array2 = my_array / my_array1  # 这里稍有争议

        my_array3 = my_array2.T
        # 获取中间总投入
        wdd = wm.iloc[aa22, aa13:aa12].values
        m2 = ["中间总投入"]
        wsm = pd.DataFrame(wdd, index=aa2, columns=m2)
        wsm1 = pd.DataFrame(wdd, index=aa2, columns=m2)

        # 分离虚拟子行业
        for i in range(cc):
            c = ws3.index[i]
            col_name = f'{prefix}_{c}'
            # ws1 = ws1.assign(**{col_name: ws.loc[c, :] * ws3.iloc[i, 1]})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 float_arr[i]
            bb = 1 - ws3.loc[c, :].values
            wsm.loc[col_name] = wsm.loc[c, :] * ws3.loc[c, :].values  # 新增行
            wsm.loc[c, :] = wsm.loc[c, :] * bb

        vb2 = wsm.iloc[0:cd, :].values  # 这是加权处理过得到数值

        CC = arr4 - vb2
        ma = CC * my_array3
        M = sum(ma)

        # for i in range(cd):
        #     C = CC[i] * my_array3[i]
        #     M += C
        # st.write(M)
        sums = M[0]  # 间接增加值
        ax = M[0]
        # 直接增加值

        direct = 0
        for i in range(cc):
            c = ws3.index[i]
            a = wd1.index.get_loc(c)
            m = wd1.iloc[a, 0] - wsm1.iloc[a, 0]
            m1 = m * ws3.loc[c, :].values
            direct += m1

        primary_industry = [
            '农产品',
            '林产品',
            '畜牧产品',
            '渔产品',
        ]

        secondary_industry = [
            '煤炭开采和洗选产品',
            '石油和天然气开采产品',
            '黑色金属矿采选产品',
            '有色金属矿采选产品',
            '非金属矿采选产品',
            '其他采矿产品',
            '农副食品加工产品',
            '食品制造产品',
            '酒、饮料和精制茶制造产品',
            '烟草制品',
            '糖及糖制品,'
            '屠宰及肉类加工品',
            '纺织服装、服饰业',
            '皮革、毛皮、羽毛及其制品和制鞋业',
            '木材加工和木、竹、藤、棕、草制品业',
            '家具制造业',
            '造纸和纸制品业',
            '印刷和记录媒介复制业'
            '文教、工美、体育和娱乐用品制造业'
            '石油、煤炭及其他燃料加工业'
            '化学原料和化学制品制造业'
            '医药制造业'
            '化学纤维制造业'
            '橡胶和塑料制品业'
            '非金属矿物制品业'
            '黑色金属冶炼和压延加工业'
            '有色金属冶炼和压延加工业'
            '金属制品业'
            '通用设备制造业'
            '专用设备制造业'
            '汽车制造业'
            '铁路、船舶、航空航天和其他运输设备制造业'
            '电气机械和器材制造业'
            '计算机、通信和其他电子设备制造业'
            '仪器仪表制造业'
            '其他制造业'
            '废弃资源综合利用业'
            '电力、热力、燃气及水生产和供应业'
            '电力、热力生产和供应业'
            '燃气生产和供应业'
            '水的生产和供应业'
            '建筑业'
            '房屋建筑业'
            '土木工程建筑业'
            '建筑安装业'
            '建筑装饰、装修和其他建筑业'
        ]

        tertiary_industry = [
            '农、林、牧、渔专业及辅助性活动',
            '开采专业及辅助性活动',
            '金属制品、机械和设备修理业',
            '批发业',
            '零售业',
            '铁路运输业',
            '道路运输业',
            '水上运输业',
            '航空运输业',
            '管道运输业',
            '多式联运和运输代理业',
            '装卸搬运和仓储业',
            '邮政业',
            '住宿业',
            '餐饮业',
            '电信、广播电视和卫星传输服务',
            '互联网和相关服务',
            '软件和信息技术服务业',
            '货币金融服务',
            '资本市场服务',
            '保险业',
            '其他金融业',
            '房地产业',
            '租赁业',
            '商务服务业',
            '研究和试验发展',
            '专业技术服务业',
            '科技推广和应用服务业',
            '水利管理业',
            '生态保护和环境治理业',
            '公共设施管理业',
            '居民服务业',
            '机动车、电子产品和日用产品修理业',
            '其他服务业',
            '教育',
            '卫生',
            '社会工作',
            '新闻和出版业',
            '广播、电视、电影和录音制作业',
            '文化艺术业',
            '体育',
            '娱乐业',
            '国家机构',
            '人民政协、民主党派',
            '社会保障',
            '群众团体、社会团体和其他成员组织',
            '基层群众自治组织及其他组织',
            '国际组织',
            '中国共产党机关'

        ]

        # 分产业核算
        df31 = pd.DataFrame(ma, index=index_values2)

        # 第一产业
        sum_values1 = 0
        for idx in df31.index:
            if idx in primary_industry:
                sum_values1 += df31.loc[idx].sum()
        # 第二产业
        sum_values2 = 0
        for idx in df31.index:
            if idx in secondary_industry:
                sum_values2 += df31.loc[idx].sum()
        # 第三产业
        sum_values3 = 0
        for idx in df31.index:
            if idx in tertiary_industry:
                sum_values3 += df31.loc[idx].sum()
    return M, direct, sum_values1, sum_values2, sum_values3


def CACULATOR():

    st.title("里昂惕夫逆矩阵计算器")
    # 生成页面标题
    local_css("style/style.css")
    # ---- LOAD ASSETS ----
    json_file_path4 = "json/animation_llq34web.json"
    with open(json_file_path4, "r") as json_file:
        lottie_coding4 = json.load(json_file)
    # 添加标题
    # 添加文字介绍
    # 欢迎

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('''
                        ##### 本页面简要说明:
                        - 本页面是个简单的里昂惕夫逆矩阵计算器，方便用户便捷计算相关值
                        - excel例表请点击下方链接获取，替换其中部门、中间投入、总产出值即可
                        - 请注意此excel表有两个sheet

                        ##### 相关功能：
                        - 直接消耗系数矩阵
                        - 列昂惕夫逆矩阵(完全消耗系数矩阵)
                        - 最终产品
                        - 计算过程相关数据下载
                        ''')

        with right_column:
            st_lottie(lottie_coding4, height=300,  key="coding6")
    file_path0 = "example/example1.xlsx"
    st.sidebar.write('<span style="color:red">tips:切换功能请重新上传对应功能所需文件</span>', unsafe_allow_html=True)

    file = st.sidebar.file_uploader("上传文件", type=["xlsx"],key=1009)

    with open(file_path0, "rb") as f:
        bytes = f.read()
        file_name = "计算示例.xlsx"
        st.download_button(
            label="点击下载示例excel文件",
            data=bytes,
            file_name=file_name,
            mime="application/xlsx",
        )
    # 创建一个Streamlit的文件上传器组件

    # 如果文件存在，读取不同的工作表并显示数据
    # 如果文件存在，读取不同的工作表并显示数据
    if file is None:
        st.warning("请在侧边栏上传Excel文件")

    else:
        # 读取Excel文件中的所有工作表名称
        sheet_names = list(pd.read_excel(file, sheet_name=None).keys())

        # 读取选定工作表的内容
        ws = pd.read_excel(file, sheet_name=sheet_names[0], index_col=0)

        # 提取索引值
        index_values = ws.index.values
        # 对读取到的数据进行处理
        # ...

        # 选择所需的数字范围
        rows = slice(0, None)  # 假设需要选择第2行到最后一行
        cols = slice(0, None)  # 假设需要选择第2列到最后一列

        # 从DataFrame对象中选择指定范围的数字，并转换为矩阵
        try:
            matrix = ws.iloc[rows, cols].values.astype(float)

            # 在这里可以进行矩阵处理和可视化
        except (ValueError, IndexError):
            st.warning("选定的数据无法计算，请在切换功能后检查并重新上传所需excel。")
        # 将数据转换为NumPy数组

        # 获取Y向量
        wd = pd.read_excel(file, sheet_name=sheet_names[1], index_col=0)

        # 确定数据范围
        rows = slice(0, None)  # 假设需要选择第2行到最后一行
        cols = slice(0, None)  # 假设需要选择第2列到最后一列
        vb = wd.iloc[rows, cols].values.astype(float)
        ab = vb.flatten()

        b = np.reciprocal(ab)  # 对数组中所有元素求倒数

        diag_mat = np.diag(b)  # 生成对角矩阵
        mat1 = np.dot(diag_mat, matrix)  # 直接消耗系数
        a = calculate_arr(mat1)

        # pd.DataFrame(a, columns=['col1', 'col2'])
        c = np.dot(a, vb)

        # 输出得到的逆矩阵
        with st.expander('直接消耗系数矩阵'):
            df = pd.DataFrame(mat1, index=index_values, columns=index_values)
            tmp_download_link = download_link(df, '直接消耗系数矩阵.csv', '点此下载')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write(df, sortable=False)

        with st.expander('列昂惕夫逆矩阵(完全消耗系数矩阵)'):
            df = pd.DataFrame(a, index=index_values, columns=index_values)
            tmp_download_link = download_link(df, '列昂惕夫逆矩阵(完全消耗系数矩阵).csv', '点此下载')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write(df, sortable=False)
        with st.expander('最终产品'):
            df2 = pd.DataFrame(c, index=index_values)

            tmp_download_link = download_link(df2, '最终产品.csv', '点此下载')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            st.write(df2, sortable=False)

        with st.expander('最终产品图表展示'):
            # 创建一个figure对象和一个axes对象
            data_flat = c.flatten()
            # 设置标题和标签
            fig = px.bar(x=data_flat[data_flat > 0], y=index_values[data_flat > 0])
            # 设置图表的高度和宽度
            fig.update_layout(height=1500, width=1000)
            # 将图形显示在Streamlit中
            st.plotly_chart(fig)


def value1():
    st.title("增加值计算（自定义）")
    # 生成页面标题
    local_css("style/style.css")
    # ---- LOAD ASSETS ----
    json_file_path4 = "json/animation_llq2t2ua.json"
    with open(json_file_path4, "r") as json_file:
        lottie_coding4 = json.load(json_file)
    # 添加标题
    # 添加文字介绍
    # 欢迎

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('''
                        ##### 本页面简要说明:
                        - 本页面使用用户自定义的投入产出表
                        - excel例表请点击下方链接获取，替换其中产业和系数值即可
                        - 本页面多数表格为交互式表格，可以点击特定进行排序which python
                        - [<下载投入产出表 >](https://data.stats.gov.cn/files/html/quickSearch/trcc/trcc08.html)

                        ##### 相关功能：
                        - 渗透增加值计算
                        - 行业贡献率计算
                        - 根据需要选取产业计算增加值
                        - 计算过程相关数据下载
                        ''')

        with right_column:
            st_lottie(lottie_coding4, height=300, key="coding5")
    # st.title('里昂惕夫逆矩阵应用')
    # st.title('里昂惕夫逆矩阵应用')
    # 生成功能介绍
    st.markdown("&nbsp;")  # 垂直间距
    keys = []
    for i in range(10):
        keys.append(str(uuid.uuid4()))

    st.sidebar.write('<span style="color:red">tips:切换功能请重新上传对应功能所需文件</span>', unsafe_allow_html=True)

    # "<h1 style='text-align: center;'>这是居中显示的文本内容</h1>", unsafe_allow_html=True)
    # 读取图像文件
    # 定义文件名和文件路径
    file_name = "example.xlsx"
    file_path0 = "example/202011.xlsx"  # 路径
    file_pathm1 = "example/20205612.xlsx"
    with open(file_pathm1, "rb") as f:
        bytes = f.read()
        file_namem = "自定义计算excel示例.xlsx"
        st.download_button(
            label="点击下载示例excel文件",
            data=bytes,
            file_name=file_namem,
            mime="application/xlsx",
        )
        # 创建一个文件上传器组件
    file1 = st.sidebar.file_uploader("上传文件", type=["xlsx"],key=154)

    # 如果文件存在，读取不同的工作表并显示数据
    if file1 is None:
        st.warning("请在侧边栏上传Excel文件")

    else:
        # 读取Excel文件中的所有工作表名称
        sheet_names = list(pd.read_excel(file1, sheet_name=None).keys())
        sheet_names1 = list(pd.read_excel(file1, sheet_name=None).keys())

        # 使用 Pandas 读取文件内容
        # 读取选定工作表的内容
        ws3 = pd.read_excel(file1, sheet_name=sheet_names1[1], index_col=0)
        # 读取选定工作表的内容

        wm = pd.read_excel(file1, sheet_name=sheet_names[0], index_col=0)

        # 比对元素是否正确
        # local of row
        aa1 = wm.iloc[0, :].values

        value_to_find = "中间使用合计"
        indices1 = np.where(aa1 == value_to_find)[0]
        try:
            aa12 = indices1[0]
            # 在这里可以进行矩阵处理和可视化
        except (ValueError, IndexError):
            st.warning("选定的数据无法计算，请在切换功能后检查并重新上传所需excel。")
        # 将数据转换为NumPy数组
        value_to_find2 = "农产品"
        indices11 = np.where(aa1 == value_to_find2)[0]
        aa13 = indices11[0]
        aa0 = aa1[aa13:aa12]

        # local of col
        aa2 = wm.iloc[:, 0].values
        value_to_find1 = "中间投入合计"
        indices2 = np.where(aa2 == value_to_find1)[0]
        aa22 = indices2[0]
        aa2 = aa2[0:aa22]
        value_to_find2 = "农产品"
        indices21 = np.where(aa2 == value_to_find2)[0]
        aa23 = indices21[0]
        aa2 = aa2[aa23:aa22]
        wdd = wm.iloc[aa13:aa12, aa23:aa22].values

        ws = pd.DataFrame(wdd, index=aa2, columns=aa2)

        ws1 = pd.DataFrame(wdd, index=aa2, columns=aa2)

        # 比对元素是否正确
        index_array = ws.index.values
        index_array1 = ws3.index.values
        set_a = set(index_array)
        set_b = set(index_array1)
        is_b_in_a = set_b.issubset(set_a)
        difference = list(set(set_b) - set(set_a))

        if is_b_in_a is False:
            st.warning("当前所上传文档中行业名称与投入产出表中行业命名不一致，请重新确认！")
            difference = list(set(set_b) - set(set_a))
            st.write("此为不一致行业:", difference)

        else:
            cc = ws3.shape[0]  # 获取相关子行业个数
            cd = ws.shape[0]  # 获取初始流量表元素个数

            # 生成加入虚拟子行业的新流量表
            prefix = '虚拟纯子行业'
            for i in range(cc):
                c = ws3.index[i]
                col_name = f'{prefix}_{c}'
                ws1 = ws1.assign(
                    **{col_name: ws1.loc[:, c] * ws3.loc[c, :].values})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 ws3.iloc[i, 1]})
                bb = 1 - ws3.loc[c, :].values
                ws1.loc[:, c] = ws1.loc[:, c] * bb  # 更新列的值
                ws1.loc[col_name] = ws1.loc[c, :] * ws3.loc[c, :].values  # 新增行

                ws1.loc[c, :] = ws1.loc[c, :] * bb

            # 获取总产量
            c1, c2 = find_indices_and_values_col(wm, "最终使用合计", "最终使用合计")

            wdd2 = wm.iloc[aa13:aa12, c2 + 2].values
            m1 = ["总产出"]
            wd = pd.DataFrame(wdd2, index=aa2, columns=m1)
            wd1 = pd.DataFrame(wdd2, index=aa2, columns=m1)

            # 生成加入虚拟子行业后的总产量表
            for i in range(cc):
                c = ws3.index[i]
                col_name = f'{prefix}_{c}'
                bb = 1 - ws3.loc[c, :].values
                wd.loc[col_name] = wd.loc[c, :] * ws3.loc[c, :].values
                # 新增行

                wd.loc[c, :] = wd.loc[c, :] * bb

            # 计算累积总和
            sw = turn(ws1)
            row_sums = np.sum(sw, axis=0)

            # 提取索引值
            index_values = ws1.index.values
            index_values2 = wd1.index.values
            # 选择所需的数字范围
            rows = slice(0, None)  # 假设需要选择第2行到最后一行
            cols = slice(0, None)  # 假设需要选择第2列到最后一列

            # 从DataFrame对象中选择指定范围的数字，并转换为矩阵
            matrix = ws1.iloc[rows, cols].values.astype(float)

            # 确定数据范围
            rows = slice(0, None)  # 假设需要选择第2行到最后一行
            cols = slice(0, None)  # 假设需要选择第2列到最后一列
            vb = wd.iloc[rows, cols].values.astype(float)
            ab = vb.flatten()
            b = np.reciprocal(ab)  # 对数组中所有元素求倒数

            # array = matrix.tolist()
            # mat1 = b * array  # 直接消耗系数
            b = np.diag(b)

            mat1 = np.dot(matrix, b)

            a = calculate_arr(mat1)  # 里昂惕夫逆矩阵

            d = calculate_arr1(mat1)  # I-A矩阵

            e = np.dot(d, vb)

            # 计算渗透值
            # 查找位置
            dff = pd.DataFrame(a, index=index_values, columns=index_values)
            # 计算累积总和
            total_df = 0
            b2 = cd - 1
            for i in range(cc):
                c = ws3.index[i]
                a = ws3.loc[c, :].values
                df = dff.iloc[0:cd, cd + i] * wd.iloc[cd + i, 0] / dff.iloc[cd + i, cd + i]
                total_df += df

            # 遍历列表中的每个 DataFrame，并将它们对应位置的值相加到 result 中
            result = total_df

            arr3 = wd.values
            my_array = np.array(result)  # 生成列向量
            arr4 = wd.iloc[0:cd, :].values  # 这是加权处理过得到数值

            my_array1 = np.array(arr4).T  # 生成列向量
            # my_array44 = wd1.iloc[0:cd, :].values
            # my_array45 = np.array(my_array44).T  # 生成列向量
            my_array2 = my_array / my_array1  # 这里稍有争议

            my_array3 = my_array2.T
            # 获取中间总投入
            # 获取中间总投入
            wdd = wm.iloc[aa22, aa13:aa12].values
            m2 = ["中间总投入"]
            wsm = pd.DataFrame(wdd, index=aa2, columns=m2)
            wsm1 = pd.DataFrame(wdd, index=aa2, columns=m2)
            # 分离虚拟子行业
            for i in range(cc):
                c = ws3.index[i]
                col_name = f'{prefix}_{c}'
                bb = 1 - ws3.loc[c, :].values
                wsm.loc[col_name] = wsm.loc[c, :] * ws3.loc[c, :].values  # 新增行
                wsm.loc[c, :] = wsm.loc[c, :] * bb

            vb2 = wsm.iloc[0:cd, :].values  # 这是加权处理过得到数值

            CC = arr4 - vb2
            ma = CC * my_array3
            M = sum(ma)

            sums = M[0]  # 间接增加值
            ax = M[0]
            # 直接增加值
            direct = 0
            for i in range(cc):
                c = ws3.index[i]
                a = wd1.index.get_loc(c)
                m = wd1.iloc[a, 0] - wsm1.iloc[a, 0]
                m1 = m * ws3.loc[c, :].values
                direct += m1

            col_name1 = ['直接增加值', '渗透增加值', '总增加值', "GDP", '贡献率']
            index1 = [""]
            # Streamlit 应用标题
            # 使用 st.number_input() 获取用户输入的 GDP 值
            gdp_value = st.number_input("请输入 GDP（万元）：", min_value=0.1)
            total_18 = direct[0] + M[0]
            per = total_18 / gdp_value
            # 在界面上显示用户输入的 GDP 值
            va = [direct[0], M[0], total_18, gdp_value, per]
            value_s = np.array(va)

            display = pd.DataFrame(value_s, index=col_name1, columns=index1)

            st.markdown("##### 增加值总览表（名义值）")
            st.markdown("###### 单位：万元")

            st.write(display)

            st.markdown("##### 自定义行业查看渗透增加值")
            m3 = ["渗透增加值"]
            wsm1 = pd.DataFrame(ma, index=aa2, columns=m3)

            # 用户选择数据项
            selected_data = st.multiselect("选择行业", aa2)
            # 用户选择图表宽度和高度

            # 用户选择图表宽度和高度
            chart_width = st.slider("选择图表宽度", min_value=100, max_value=1000, value=600)
            chart_height = st.slider("选择图表高度", min_value=100, max_value=800, value=400)

            # 根据用户选择过滤数据
            filtered_df = wsm1.loc[selected_data]

            # 使用 Altair 创建交互式图表
            chart = alt.Chart(filtered_df.reset_index()).mark_bar().encode(
                x='index:N',
                y='渗透增加值:Q'
            ).properties(
                width=chart_width,
                height=chart_height
            )

            st.altair_chart(chart)
            df31 = pd.DataFrame(ma, index=index_values2)
            st.markdown('##### 自选行业加总渗透增加值')
            selected_industries = st.multiselect('', df31.index.tolist(), key=12345)
            if selected_industries:
                selected_df = df31.loc[selected_industries]
                total_values = selected_df.sum().values
                st.markdown(f"###### 选中行业分类的总值为：{total_values}")

            # 使用loc方法获取选中数据的值

            # 根据选中的行业分类计算总值

            # 修改列名
            st.markdown("### 数据下载")

            with st.expander('加入纯虚拟子行业后的流量表'):
                tmp_download_link = download_link(ws1, '加入纯虚拟子行业后的流量表.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(ws1, sortable=False)
                # 输出得到的逆矩阵

            with st.expander('加入纯虚拟子行业后的总产品'):
                tmp_download_link = download_link(wd, '加入纯虚拟子行业后的总产品.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(wd, sortable=False)

            with st.expander('加入纯虚拟子行业后的直接消耗系数矩阵'):
                df2 = pd.DataFrame(mat1, index=index_values, columns=index_values)
                tmp_download_link = download_link(df2, '加入纯虚拟子行业后的直接消耗系数矩阵.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df2, sortable=False)
                #

            with st.expander('加入纯虚拟子行业后的列昂惕夫逆矩阵'):
                df = pd.DataFrame(a, index=index_values, columns=index_values)
                tmp_download_link = download_link(df, '加入纯虚拟子行业后的列昂惕夫逆矩阵.csv',
                                                  '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(dff, sortable=False)

            with st.expander('所求行业直接相关行业总投入对其他行业总产出的贡献'):
                df3 = pd.DataFrame(result, index=index_values2)
                tmp_download_link = download_link(df3, '工业互联网直接行业总投入对其他行业总产出的贡献.csv',
                                                  '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df3, sortable=False)

            with st.expander('所求行业对其他行业的定向贡献比'):
                df3 = pd.DataFrame(my_array2.T, index=index_values2)
                tmp_download_link = download_link(df3, '所求行业对其他行业的定向贡献比.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df3, sortable=False)
                # 创建一个figure对象和一个axes对象
                data_flat = my_array2.T.flatten()
                # 设置标题和标签
                fig = px.bar(x=data_flat[data_flat > 0], y=index_values2[data_flat > 0])
                # 设置图表的高度和宽度
                fig.update_layout(height=1500, width=1000)
                # 将图形显示在Streamlit中
                st.plotly_chart(fig)

            with st.expander('对各行业渗透增加值'):
                df3 = pd.DataFrame(ma, index=index_values2)
                tmp_download_link = download_link(df3, '渗透增加值.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                # Streamlit 应用
                st.write(df3)


def Value():
    st.title("增加值计算（内置）")
    # 生成页面标题
    local_css("style/style.css")
    # ---- LOAD ASSETS ----
    json_file_path3 = "json/animation_llq2osgj (1).json"
    with open(json_file_path3, "r") as json_file:
        lottie_coding3 = json.load(json_file)
    # 添加标题
    # 添加文字介绍
    # 欢迎

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown('''
                        ##### 本页面简要说明:
                        - 本页面使用内置的投入产出表，您只需要提供相关系数excel表
                        - excel例表请点击下方链接获取，替换其中产业和系数值即可
                        - 本页面多数表格为交互式表格，可以点击特定进行排序

                        ##### 相关功能：
                        - 渗透增加值计算
                        - 行业贡献率计算
                        - 根据需要选取产业计算增加值
                        - 计算过程相关数据下载
                        ''')

            # 显示欢迎文本

        with right_column:
            st_lottie(lottie_coding3, height=300, key="coding")
    # st.title('里昂惕夫逆矩阵应用')
    # 生成功能介绍


    st.markdown("&nbsp;")  # 垂直间距
    # 在主要内容区域中显示输入的文本
    # 生成唯一的密钥
    keys = []
    for i in range(10):
        keys.append(str(uuid.uuid4()))

    st.sidebar.write('<span style="color:red">tips:切换功能请重新上传对应功能所需文件</span>', unsafe_allow_html=True)

    # "<h1 style='text-align: center;'>这是居中显示的文本内容</h1>", unsafe_allow_html=True)
    # 读取图像文件
    # 定义文件名和文件路径
    file_name = "example.xlsx"
    file_path0 = "example/202011.xlsx"  # 路径
    file_path8 = "example/2018测试.xlsx"
    file_pathe = "example/相关系数.xlsx"

    # with st.expander('示例图'):
    #     st.write(
    #         "需要注意的是每个sheet的内容和对应数据，sheet1放部门流量表，sheet2放总产出表，sheet3放中间总投入，sheet4放置相关系数")
    #     # 定义文件名和文件路径

    with open(file_pathe, "rb") as f:
        bytes = f.read()
        file_nameee = "增加值计算示例excel.xlsx"
        st.download_button(
            label="点击下载示例excel文件",
            data=bytes,
            file_name=file_nameee,
            mime="application/xlsx",
        )
    option = st.sidebar.selectbox(
        '请选择的年份:',
        ('2018', '2020')
    )
    if option == "2018":
        file_path = "example/2018测试.xlsx"
    else:
        file_path = "example/202011.xlsx"

    # 创建一个文件上传器组件
    file1 = st.sidebar.file_uploader("上传文件", type=["xlsx"],key=1594)

    # 如果文件存在，读取不同的工作表并显示数据
    if file1 is None:
        st.warning("请在侧边栏上传Excel文件")

    else:
        # 读取Excel文件中的所有工作表名称
        sheet_names = list(pd.read_excel(file_path, sheet_name=None).keys())
        sheet_names1 = list(pd.read_excel(file1, sheet_name=None).keys())

        # 使用 Pandas 读取文件内容
        # 读取选定工作表的内容
        ws3 = pd.read_excel(file1, sheet_name=sheet_names1[0], index_col=0)

        wm = pd.read_excel(file_path, sheet_name=sheet_names[0], index_col=0)

        # 比对元素是否正确
        # local of row
        aa1 = wm.iloc[0, :].values

        value_to_find = "中间使用合计"
        indices1 = np.where(aa1 == value_to_find)[0]
        aa12 = indices1[0]
        value_to_find2 = "农产品"
        indices11 = np.where(aa1 == value_to_find2)[0]
        aa13 = indices11[0]
        aa0 = aa1[aa13:aa12]

        # local of col
        aa2 = wm.iloc[:, 0].values
        value_to_find1 = "中间投入合计"
        indices2 = np.where(aa2 == value_to_find1)[0]
        aa22 = indices2[0]
        aa2 = aa2[0:aa22]
        value_to_find2 = "农产品"
        indices21 = np.where(aa2 == value_to_find2)[0]
        aa23 = indices21[0]
        aa2 = aa2[aa23:aa22]
        wdd = wm.iloc[aa13:aa12, aa23:aa22].values

        ws = pd.DataFrame(wdd, index=aa2, columns=aa2)

        ws1 = pd.DataFrame(wdd, index=aa2, columns=aa2)

        index_array = ws.index.values
        index_array1 = ws3.index.values
        set_a = set(index_array)
        set_b = set(index_array1)
        is_b_in_a = set_b.issubset(set_a)
        difference = list(set(set_b) - set(set_a))

        if is_b_in_a is False:
            st.warning("当前所上传文档中行业名称与投入产出表中行业命名不一致，请重新确认！")
            difference = list(set(set_b) - set(set_a))
            st.write("此为不一致行业:", difference)

        else:

            cc = ws3.shape[0]  # 获取相关子行业个数
            cd = ws.shape[0]  # 获取初始流量表元素个数

            # 生成加入虚拟子行业的新流量表
            prefix = '虚拟纯子行业'
            for i in range(cc):
                c = ws3.index[i]
                col_name = f'{prefix}_{c}'
                ws1 = ws1.assign(
                    **{col_name: ws1.loc[:, c] * ws3.loc[c, :].values})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 ws3.iloc[i, 1]})
                bb = 1 - ws3.loc[c, :].values
                ws1.loc[:, c] = ws1.loc[:, c] * bb  # 更新列的值
                try:
                    ws1.loc[col_name] = ws1.loc[c, :] * ws3.loc[c, :].values  # 新增行
                except (ValueError, IndexError):
                    st.warning("选定的数据无法计算，请在切换功能后检查并重新上传所需excel。")
                ws1.loc[c, :] = ws1.loc[c, :] * bb

            # 获取总产量
            c1, c2 = find_indices_and_values_col(wm, "最终使用合计", "最终使用合计")

            wdd2 = wm.iloc[aa13:aa12, c2 + 2].values
            m1 = ["总产出"]
            wd = pd.DataFrame(wdd2, index=aa2, columns=m1)

            wd1 = pd.DataFrame(wdd2, index=aa2, columns=m1)

            # 生成加入虚拟子行业后的总产量表
            for i in range(cc):
                c = ws3.index[i]
                col_name = f'{prefix}_{c}'
                # ws1 = ws1.assign(**{col_name: ws.loc[c, :] * ws3.iloc[i, 1]})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 float_arr[i]
                bb = 1 - ws3.loc[c, :].values
                wd.loc[col_name] = wd.loc[c, :] * ws3.loc[c, :].values

                # 新增行
                wd.loc[c, :] = wd.loc[c, :] * bb

            # 计算累积总和
            sw = turn(ws1)
            row_sums = np.sum(sw, axis=0)

            # 提取索引值
            index_values = ws1.index.values
            index_values2 = wd1.index.values
            # 选择所需的数字范围
            rows = slice(0, None)  # 假设需要选择第2行到最后一行
            cols = slice(0, None)  # 假设需要选择第2列到最后一列

            # 从DataFrame对象中选择指定范围的数字，并转换为矩阵
            matrix = ws1.iloc[rows, cols].values.astype(float)

            # 确定数据范围
            rows = slice(0, None)  # 假设需要选择第2行到最后一行
            cols = slice(0, None)  # 假设需要选择第2列到最后一列
            vb = wd.iloc[rows, cols].values.astype(float)
            ab = vb.flatten()
            b = np.reciprocal(ab)  # 对数组中所有元素求倒数

            # array = matrix.tolist()
            # mat1 = b * array  # 直接消耗系数
            b = np.diag(b)

            mat1 = np.dot(matrix, b)

            a56 = calculate_arr(mat1)  # 里昂惕夫逆矩阵

            d = calculate_arr1(mat1)  # I-A矩阵

            e = np.dot(d, vb)

            # 计算渗透值
            # 查找位置
            dff = pd.DataFrame(a56, index=index_values, columns=index_values)
            # 计算累积总和
            total_df = 0
            b2 = cd - 1
            for i in range(cc):
                c = ws3.index[i]
                a = ws3.loc[c, :].values
                df = dff.iloc[0:cd, cd + i] * wd.iloc[cd + i, 0] / dff.iloc[cd + i, cd + i]
                total_df += df

            # 遍历列表中的每个 DataFrame，并将它们对应位置的值相加到 result 中
            result = total_df

            arr3 = wd.values
            my_array = np.array(result)  # 生成列向量
            arr4 = wd.iloc[0:cd, :].values  # 这是加权处理过得到数值

            my_array1 = np.array(arr4).T  # 生成列向量
            # my_array44 = wd1.iloc[0:cd, :].values
            # my_array45 = np.array(my_array44).T  # 生成列向量
            my_array2 = my_array / my_array1  # 这里稍有争议

            my_array3 = my_array2.T
            # 获取中间总投入
            wdd = wm.iloc[aa22, aa13:aa12].values
            m2 = ["中间总投入"]
            wsm = pd.DataFrame(wdd, index=aa2, columns=m2)
            wsm1 = pd.DataFrame(wdd, index=aa2, columns=m2)

            # 分离虚拟子行业
            for i in range(cc):
                c = ws3.index[i]
                col_name = f'{prefix}_{c}'
                # ws1 = ws1.assign(**{col_name: ws.loc[c, :] * ws3.iloc[i, 1]})  # 为数据框新增列，并将该列的所有元素赋值为 x 列的值乘以 float_arr[i]
                bb = 1 - ws3.loc[c, :].values
                wsm.loc[col_name] = wsm.loc[c, :] * ws3.loc[c, :].values  # 新增行
                wsm.loc[c, :] = wsm.loc[c, :] * bb

            vb2 = wsm.iloc[0:cd, :].values  # 这是加权处理过得到数值

            CC = arr4 - vb2
            ma = CC * my_array3
            M = sum(ma)

            # for i in range(cd):
            #     C = CC[i] * my_array3[i]
            #     M += C
            # st.write(M)
            sums = M[0]  # 间接增加值
            ax = M[0]
            # 直接增加值

            direct = 0
            for i in range(cc):
                c = ws3.index[i]
                a = wd1.index.get_loc(c)
                m = wd1.iloc[a, 0] - wsm1.iloc[a, 0]
                m1 = m * ws3.loc[c, :].values
                direct += m1

            primary_industry = [
                '农产品',
                '林产品',
                '畜牧产品',
                '渔产品',
            ]

            secondary_industry = [
                '煤炭开采和洗选产品',
                '石油和天然气开采产品',
                '黑色金属矿采选产品',
                '有色金属矿采选产品',
                '非金属矿采选产品',
                '其他采矿产品',
                '农副食品加工产品',
                '食品制造产品',
                '酒、饮料和精制茶制造产品',
                '烟草制品',
                '糖及糖制品,'
                '屠宰及肉类加工品',
                '纺织服装、服饰业',
                '皮革、毛皮、羽毛及其制品和制鞋业',
                '木材加工和木、竹、藤、棕、草制品业',
                '家具制造业',
                '造纸和纸制品业',
                '印刷和记录媒介复制业'
                '文教、工美、体育和娱乐用品制造业'
                '石油、煤炭及其他燃料加工业'
                '化学原料和化学制品制造业'
                '医药制造业'
                '化学纤维制造业'
                '橡胶和塑料制品业'
                '非金属矿物制品业'
                '黑色金属冶炼和压延加工业'
                '有色金属冶炼和压延加工业'
                '金属制品业'
                '通用设备制造业'
                '专用设备制造业'
                '汽车制造业'
                '铁路、船舶、航空航天和其他运输设备制造业'
                '电气机械和器材制造业'
                '计算机、通信和其他电子设备制造业'
                '仪器仪表制造业'
                '其他制造业'
                '废弃资源综合利用业'
                '电力、热力、燃气及水生产和供应业'
                '电力、热力生产和供应业'
                '燃气生产和供应业'
                '水的生产和供应业'
                '建筑业'
                '房屋建筑业'
                '土木工程建筑业'
                '建筑安装业'
                '建筑装饰、装修和其他建筑业'
            ]

            tertiary_industry = [
                '农、林、牧、渔专业及辅助性活动',
                '开采专业及辅助性活动',
                '金属制品、机械和设备修理业',
                '批发业',
                '零售业',
                '铁路运输业',
                '道路运输业',
                '水上运输业',
                '航空运输业',
                '管道运输业',
                '多式联运和运输代理业',
                '装卸搬运和仓储业',
                '邮政业',
                '住宿业',
                '餐饮业',
                '电信、广播电视和卫星传输服务',
                '互联网和相关服务',
                '软件和信息技术服务业',
                '货币金融服务',
                '资本市场服务',
                '保险业',
                '其他金融业',
                '房地产业',
                '租赁业',
                '商务服务业',
                '研究和试验发展',
                '专业技术服务业',
                '科技推广和应用服务业',
                '水利管理业',
                '生态保护和环境治理业',
                '公共设施管理业',
                '居民服务业',
                '机动车、电子产品和日用产品修理业',
                '其他服务业',
                '教育',
                '卫生',
                '社会工作',
                '新闻和出版业',
                '广播、电视、电影和录音制作业',
                '文化艺术业',
                '体育',
                '娱乐业',
                '国家机构',
                '人民政协、民主党派',
                '社会保障',
                '群众团体、社会团体和其他成员组织',
                '基层群众自治组织及其他组织',
                '国际组织',
                '中国共产党机关'

            ]

            # 分产业核算
            df31 = pd.DataFrame(ma, index=index_values2)

            # 第一产业
            sum_values1 = 0
            for idx in df31.index:
                if idx in primary_industry:
                    sum_values1 += df31.loc[idx].sum()
            # 第二产业
            sum_values2 = 0
            for idx in df31.index:
                if idx in secondary_industry:
                    sum_values2 += df31.loc[idx].sum()
            # 第三产业
            sum_values3 = 0
            for idx in df31.index:
                if idx in tertiary_industry:
                    sum_values3 += df31.loc[idx].sum()

            # Streamlit 应用

            # Streamlit 应用

            # # 用户输入参数
            # a = st.number_input("Enter value for First Industry (a)", min_value=0, step=1)
            # b = st.number_input("Enter value for Second Industry (b)", min_value=0, step=1)
            # c = st.number_input("Enter value for Third Industry (c)", min_value=0, step=1)

            # 示例数据
            industries = ['First Industry', 'Second Industry', 'Third Industry']
            values = [sum_values1, sum_values2, sum_values3]

            # Streamlit 应用
            if option == "2018":
                file_path = file_path8
            else:
                file_path = file_path0

            try:
                aq1, bq1, cq1, dq1, eq1 = caler(file_path, file1)
            except Exception as e:
                aq1, bq1, cq1, dq1, eq1 = None, None, None, None, None

            # 输出计算结果或错误信息
            if aq1 is not None:
                aq, bq, cq, dq, eq = caler(file_path8, file1)
                aq2, bq2, cq2, dq2, eq2 = caler(file_path0, file1)

                # 计算总值和贡献率
                gdp18 = 9003090000
                gdp20 = 10135670000
                total_18 = bq[0] + aq[0]
                total_20 = (bq2[0] + aq2[0])
                per18 = total_18 / gdp18
                per20 = total_20 / gdp20

                col_name1 = ['直接增加值', '渗透增加值', '总增加值', "GDP", '贡献率']
                index1 = ["2018", "2020"]
                value_s = np.array(
                    [[bq[0], aq[0], total_18, gdp18, per18], [bq2[0] , aq2[0] , total_20, gdp20, per20]])

                display = pd.DataFrame(value_s, index=index1, columns=col_name1)
                st.markdown("##### 增加值总览表")
                st.markdown("###### 单位：万元")
                st.markdown("###### 名义值")

                st.write(display)

                data1 = {
                    'Year': ['2018', '2018', '2020', '2020'],
                    'Industry': ['直接增加值', '渗透增加值', '直接增加值', '渗透增加值'],
                    'Value': [bq[0], aq[0], bq2[0], aq2[0]]
                }

                df1 = pd.DataFrame(data1)

                # Streamlit 应用

                # 创建堆积柱状图
                fig = px.bar(df1, x='Year', y='Value', color='Industry', barmode='stack')

                # 设置图表的布局
                fig.update_layout(title='堆积柱状图比较不同年份增加值构成', xaxis_title='Year',
                                  yaxis_title='Value')

                # 将图形显示在 Streamlit 中
                st.plotly_chart(fig)

                data = {
                    'Year': ['2018', '2018', '2018', '2020', '2020', '2020'],
                    'Industry': ['First Industry', 'Second Industry', 'Third Industry', 'First Industry',
                                 'Second Industry', 'Third Industry'],
                    'Value': [cq, dq, eq, cq2, dq2, eq2]
                }

                df = pd.DataFrame(data)

                # Streamlit 应用

                # 创建堆积柱状图
                fig = px.bar(df, x='Year', y='Value', color='Industry', barmode='stack')

                # 设置图表的布局
                fig.update_layout(title='不同年份渗透增加值堆积柱状图', xaxis_title='Year',
                                  yaxis_title='Value Added')

                # 将图形显示在 Streamlit 中
                st.plotly_chart(fig)
                industries = ['第一产业', '第二产业', '第三产业']
                values = [sum_values1, sum_values2, sum_values3]

                fig1 = px.pie(names=industries, values=values, color=industries)
                fig1.update_layout(
                    title='2018年分产业渗透增加值饼图',  # 设置标题
                    # title_x=0.5  # 标题的水平位置，默认为0.5（居中）
                )

                local_css("style/style.css")

                values = [cq2, dq2, eq2]
                fig2 = px.pie(names=industries, values=values, color=industries)
                fig2.update_layout(
                    title='2020年分产业渗透增加值饼图',  # 设置标题
                    # title_x=0.5  # 标题的水平位置，默认为0.5（居中）
                )
                with st.container():
                    st.write("---")
                    left_column, right_column = st.columns(2)
                    with left_column:
                        st.plotly_chart(fig1)

                    with right_column:
                        st.plotly_chart(fig2)


                st.markdown("##### 自定义行业查看 {}年渗透增加值".format(option))
                m3 = ["渗透增加值"]
                wsm1 = pd.DataFrame(ma, index=aa2, columns=m3)

                # 用户选择数据项
                selected_data = st.multiselect("选择行业", aa2)
                # 用户选择图表宽度和高度

                # 用户选择图表宽度和高度
                chart_width = st.slider("选择图表宽度", min_value=100, max_value=1000, value=600)
                chart_height = st.slider("选择图表高度", min_value=100, max_value=800, value=400)

                # 根据用户选择过滤数据
                filtered_df = wsm1.loc[selected_data]

                # 使用 Altair 创建交互式图表
                chart = alt.Chart(filtered_df.reset_index()).mark_bar().encode(
                    x='index:N',
                    y='渗透增加值:Q'
                ).properties(
                    width=chart_width,
                    height=chart_height
                )

                st.altair_chart(chart)


            else:
                # 示例数据
                industries = ['第一产业', '第二产业', '第三产业']
                values = [sum_values1, sum_values2, sum_values3]
                # Streamlit 应用
                # 创建 Plotly 条形图
                fig = px.bar(x=industries, y=values, color=industries)

                # 设置图表的布局
                fig.update_layout(title=' 分产业大类渗透增加值', xaxis_title='Industries',
                                  yaxis_title='渗透增加值')

                # 将图形显示在 Streamlit 中
                st.plotly_chart(fig)
                fig1 = px.pie(names=industries, values=values, color=industries)
                st.plotly_chart(fig1)

            st.markdown('##### 自选行业加总渗透增加值')
            selected_industries = st.multiselect('', df31.index.tolist(), key=12345)
            if selected_industries:
                selected_df = df31.loc[selected_industries]
                total_values = selected_df.sum().values
                st.markdown(f"###### 选中行业分类的总值为：{total_values}")

            # 使用loc方法获取选中数据的值

            # 根据选中的行业分类计算总值

            # 修改列名
            st.markdown("### 数据下载")

            with st.expander('加入纯虚拟子行业后的流量表'):
                tmp_download_link = download_link(ws1, '加入纯虚拟子行业后的流量表.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(ws1, sortable=False)
                # 输出得到的逆矩阵

            with st.expander('加入纯虚拟子行业后的总产品'):
                tmp_download_link = download_link(wd, '加入纯虚拟子行业后的总产品.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(wd, sortable=False)

            with st.expander('加入纯虚拟子行业后的直接消耗系数矩阵'):
                df2 = pd.DataFrame(mat1, index=index_values, columns=index_values)
                tmp_download_link = download_link(df2, '加入纯虚拟子行业后的直接消耗系数矩阵.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df2, sortable=False)
                #
            with st.expander('加入纯虚拟子行业后的列昂惕夫逆矩阵'):
                df = pd.DataFrame(a56, index=index_values, columns=index_values)
                tmp_download_link = download_link(df, '加入纯虚拟子行业后的列昂惕夫逆矩阵.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df, sortable=False)

            # with st.expander('加入纯虚拟子行业后的列昂惕夫逆矩阵'):
            #     df = pd.DataFrame(a56, index=index_values, columns=index_values)
            #     tmp_download_link = download_link(df, '加入纯虚拟子行业后的列昂惕夫逆矩阵.csv',
            #                                       '点此下载')
            #     st.markdown(tmp_download_link, unsafe_allow_html=True)
            #     st.write(df, sortable=False)

            with st.expander('所求行业直接相关行业总投入对其他行业总产出的贡献'):
                df3 = pd.DataFrame(result, index=index_values2)
                tmp_download_link = download_link(df3, '工业互联网直接行业总投入对其他行业总产出的贡献.csv',
                                                  '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df3, sortable=False)

            with st.expander('所求行业对其他行业的定向贡献比'):
                df3 = pd.DataFrame(my_array2.T, index=index_values2)
                tmp_download_link = download_link(df3, '所求行业对其他行业的定向贡献比.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                st.write(df3, sortable=False)
                # 创建一个figure对象和一个axes对象
                data_flat = my_array2.T.flatten()
                # 设置标题和标签
                fig = px.bar(x=data_flat[data_flat > 0], y=index_values2[data_flat > 0])
                # 设置图表的高度和宽度
                fig.update_layout(height=1500, width=1000)
                # 将图形显示在Streamlit中
                st.plotly_chart(fig)

            with st.expander('对各行业渗透增加值'):
                df3 = pd.DataFrame(ma, index=index_values2)
                tmp_download_link = download_link(df3, '渗透增加值.csv', '点此下载')
                st.markdown(tmp_download_link, unsafe_allow_html=True)
                # Streamlit 应用
                st.write(df3)


# Calculator Page
def home_page():
    local_css("style/style.css")
    # ---- LOAD ASSETS ----
    json_file_path = "json/animation_llq1nx9e.json"
    with open(json_file_path, "r") as json_file:
        lottie_coding = json.load(json_file)

    json_file_path2 = "json/animation_llq2g8pc.json"
    with open(json_file_path2, "r") as json_file:
        lottie_coding2 = json.load(json_file)

    # 添加标题
    st.title("欢迎您使用本程序！")

    # 添加文字介绍
    # 欢迎

    with st.container():
        st.write("---")
        left_column, right_column = st.columns(2)
        with left_column:
            st.header("What I do")
            welcome_text = """
               本程序的主要功能是通过投入产出分析方法来计算行业的渗透增加值。这种方法参考了《中国工业互联网产业经济发展白皮书（2021 年）》中提供的渗透增加值算法。基于计算结果，本程序还提供了一些可视化图表，以帮助您更直观地理解结果。

               本程序都旨在为您提供简单易用的界面。您可以轻松输入所需数据并获取计算结果，同时也可以通过图表分析来更好地理解数据。

               我们相信这个程序将为您的分析工作带来便捷和可视化的体验。请随意使用，并享受它带来的便利！
               """

            # 显示欢迎文本
            st.markdown(welcome_text)

        with right_column:
            st_lottie(lottie_coding, height=300, key="coding")
    # Sphere
    with st.container():
        st.write("---")
        st.header("使用教程")
        st.write("##")
        image_column, text_column = st.columns(2)
        with image_column:
            st_lottie(lottie_coding2,  key="coding2")
        with text_column:
            st.write(
                """
                在开始程序使用之前，请快速浏览本教程，以便于正确操作
                
                下方提供教程的下载链接
                """
            )

            file_path = 'pdf/《中国工业互联网产业经济发展白皮书（2021年）》_57-60.pdf'
            with open(file_path, 'rb') as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()

            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="计算原理.pdf">点击这里下载计算原理文档</a>',
                unsafe_allow_html=True)

            file_path = 'pdf/帮助文档.pdf'
            with open(file_path, 'rb') as f:
                data = f.read()
                b64 = base64.b64encode(data).decode()

            st.markdown(
                f'<a href="data:application/pdf;base64,{b64}" download="帮助文档.pdf">点击这里下载帮助文档</a>',
                unsafe_allow_html=True)


# Main App
def main():
    st.set_page_config(page_title="MyApp", page_icon=":tiger:", layout="wide")
    json_file_path1 = "json/animation_llq1zf9c.json"
    with open(json_file_path1, "r") as json_file:
        lottie_coding1 = json.load(json_file)

    img_sphere = Image.open("images/sphere.jpg")
    # 设置页面标题和布局
    st_lottie(lottie_coding1, height=100, key="coding1")

    pages = {
        "主页": home_page,
        "增加值相关计算(作者提供)": Value,
        "增加值计算(自定义投入产出表)": value1,
        "里昂惕夫逆矩阵计算": CACULATOR
    }




    selection = st.sidebar.radio("功能切换:", list(pages.keys()))
    page = pages[selection]
    page()


if __name__ == "__main__":
    main()
