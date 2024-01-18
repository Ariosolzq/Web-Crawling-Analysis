from urllib.parse import urlencode
import requests
from lxml import etree
import time
import csv
import re
from pyquery import PyQuery as pq

urls = []  # 详情页url
names = []  # 房屋标签
apartments = []  # 小区
neighbourhoods = []  # 社区
introductions = []  # 基本信息
regions = []  # 行政区，详情页获取
house_type = []  # 户型，拆分
house_square = []  # 面积，拆分
house_direction = []  # 朝向，拆分
furniture = []  # 装修类型，拆分
heated = []  # 房屋热度
likes = []  # 关注人数，拆分
times = []  # 发布时间，拆分
prices = []  # 总价
per_prices = []  # 单价

page = 1
base_url = "https://bj.lianjia.com/ershoufang/pg{}/?".format(page)
data = {
    "type": 1,
    'query': 'https://bj.lianjia.com/ershoufang/pg{}/'.format(page)
}
headers = {
    "cookie": "uuid_tt_dd=10_37481457760-1614696081762-771893; UN=Smart_look; Hm_ct_6bcd52f51e9b3dce32bec4a3997715ac=6525*1*10_37481457760-1614696081762-771893!5744*1*Smart_look; Hm_lvt_e5ef47b9f471504959267fd614d579cd=1620037344; __gads=ID=b2b2ba9fbd5f97fa-228d5d7c93cb004f:T=1631355448:RT=1631355448:S=ALNI_MaN-YuGZ0ZKrwQlnce2q2uTVYcQrQ; ssxmod_itna=Yq0h0K4Axx7tGH3iQ0gxQqAITqyDUEhTq874Y5dD/I3xnqD=GFDK40oAOIKuIx==YD8aiQFnADcfhq81RgxWd5n7cjxqGCe0aDbqGk7nzx4GGUxBYDQxAYDGDDPDoxKD1D3qDkD7g1ZlndtDm4GWzqGfDDoDYS6nDitD4qDB+2dDKqGgzwrXGeNQ6qqKv7+yD0t=xBd=F6hyAmaHk0NKocWjDxDHf8yo8Ah1rg443CpnDB6mxBQZ0MN00CHgDCX4=rhaKixYSEG69z4rKDPqeb4T5hD=KBDi9S4W/DqudE+zR5DADU44QOD4D===; ssxmod_itna2=Yq0h0K4Axx7tGH3iQ0gxQqAITqyDUEhTq874Y5G9Wv7DBTkAx7p+x8=QnhyoS8GlrMviCP5KQo=wDO/4w=85G3piamrVr=3th8s9p7XsBA/TE4+NN/l5R9W2Ee=Mc4LNE2L8YLdjNkUS=+U3i=IGXcIB3D8EUTWl5WzEr4Ar=480rZRov3v7d8CePQv4n8+bA5TZoFEraWhcaiULRdZfbcifrcWfr0AfHvFaq+LCtHVqnwsnoQbjKvvA5vLOD=WDO88nQPx9aCSFOUucO0H=3nOzCDgIH2=Hl67yls8vgsvVoEdkX/2eKWncBDu/xx7m9Dvlm+00hZIaSQpM+a=tDVIBifPzQcOI+kle5wcNWwM3E+Ww5=IhSBitgc07P6PQ15b4W=62VIg8I2b0oKtG3D07UYb1bxh0K8AN30AmQE7M8qGQcGqcSqHbxsGQV0f08T97DWSA+PFT07Sok/GmpP8KkfqfK=Z+YZ+YG8LPTqGUNYjADDLxD2/yWMgPBq4l2xnFqAKzDhKBqYKmKAnI0TAS2Y0QzkDPFdnBuzgTIZ2DDWTZhKQexe7IPD==; UserName=Smart_look; UserInfo=d43e50a1aa0c4f2caa52662c49ea3f55; UserToken=d43e50a1aa0c4f2caa52662c49ea3f55; UserNick=Amae; AU=779; BT=1635603087677; p_uid=U010000; Hm_up_6bcd52f51e9b3dce32bec4a3997715ac=%7B%22islogin%22%3A%7B%22value%22%3A%221%22%2C%22scope%22%3A1%7D%2C%22isonline%22%3A%7B%22value%22%3A%221%22%2C%22scope%22%3A1%7D%2C%22isvip%22%3A%7B%22value%22%3A%220%22%2C%22scope%22%3A1%7D%2C%22uid_%22%3A%7B%22value%22%3A%22Smart_look%22%2C%22scope%22%3A1%7D%7D; c_segment=5; dc_sid=8af578122c748c6f8f79293ff5d9c725; c_first_ref=www.baidu.com; c_first_page=https%3A//blog.csdn.net/quanqxj/article/details/89226160; Hm_lvt_6bcd52f51e9b3dce32bec4a3997715ac=1640932419,1640958689,1641047085,1641108277; c_pref=https%3A//www.baidu.com/link; c_ref=https%3A//blog.csdn.net/Smart_look%3Fspm%3D1000.2115.3001.5343; firstDie=1; log_Id_click=32; Hm_lpvt_6bcd52f51e9b3dce32bec4a3997715ac=1641188306; log_Id_view=97; dc_session_id=10_1641198463093.943635; c_page_id=default; dc_tos=r54ku6; log_Id_pv=73",
    "referer": "https://bj.lianjia.com/ershoufang/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/96.0.4664.45 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}


# 控制访问页数
def page_control(i):
    global page, base_url, headers, data
    page = i

    base_url = "https://bj.lianjia.com/ershoufang/pg{}/?".format(page)

    data = {
        "type": 1,
        'query': 'https://bj.lianjia.com/ershoufang/pg{}/'.format(page)
    }
    return 0


# 爬取内容
def get_content(base_url, headers, data):
    url = base_url + urlencode(data)
    global urls, names, apartments, neighbourhoods, introductions, heated, prices, per_prices

    try:
        res = requests.get(url, headers=headers)
        time.sleep(2.5)
        if res.status_code == 200:
            # res.encoding = 'utf-8'
            res.encoding = 'gbk2312'
            res_text = res.text
            soup = etree.HTML(res_text)

            urls = soup.xpath('//*[ @ log-mod = "list"]/li/a/@href')
            names = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[1]/a/text()')
            apartments = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[2]/div/a[1]/text()')
            neighbourhoods = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[2]/div/a[2]/text()')
            introductions = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[3]/div/text()')
            heated = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[4]/text()')
            prices = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[6]/div[1]/span/text()')
            per_prices = soup.xpath('//*[ @ log-mod = "list"]/li/div[1]/div[6]/div[2]/span/text()')

            print(urls[0], names[0], apartments[0], neighbourhoods[0], introductions[0], heated[0], prices[0],
                  per_prices[0])
            return 0

    except requests.ConnectionError as e:
        print("Error", e.args)
        return 0


# 获取行政区
def get_region():
    global regions
    for url_region in urls:
        res_region = requests.get(url_region)
        time.sleep(1)
        res_region.encoding = 'gbk2312'
        res_region_text = res_region.text
        soup = etree.HTML(res_region_text)
        regions_item = soup.xpath('/html/body/div[5]/div[2]/div[5]/div[2]/span[2]/a[1]/text()')
        regions.append(regions_item[0])
    return 0


# 拆分基本信息
def split_info():
    global house_type, house_direction, furniture, house_square
    for info in introductions:
        item = info.split(' | ')
        # print(info)
        house_type.append(item[0])
        house_direction.append(item[2])
        furniture.append(item[3])
        square = item[1].split('平米')
        house_square.append(square[0])
    return 0


# 拆分热度
def split_heated():
    global likes, times
    for info in heated:
        item = info.split(' / ')
        # print(item)
        like = item[0].split('人')
        likes.append(like[0])
        s1='天'
        s2='个月'
        if s1 in item[1]:
            time_split=item[1].split(s1)
            times.append(time_split[0])
        elif s2 in item[1]:
            time_split = item[1].split(s2)
            times_count=int(time_split[0])*30
            times.append(str(times_count))
        else: times.append('365')
    return 0


# 保存结果
def save_csv():
    global names, apartments, neighbourhoods, regions, house_type, house_square, house_direction, furniture
    global times, prices, per_prices
    for j in range(0, len(names)):
        write.writerow({'房屋标签': names[j], '小区': apartments[j], '社区': neighbourhoods[j], '行政区': regions[j],
                        '户型': house_type[j], '面积': house_square[j], '朝向': house_direction[j], '装修类型': furniture[j],
                        '关注人数': likes[j], '发布时间': times[j], '总价': prices[j], '平方米价格': per_prices[j]})


def clear_list():
    global urls, names, apartments, neighbourhoods, introductions, heated, prices, per_prices
    global regions, house_type, house_direction, furniture, house_square, likes, times
    urls = []  # 详情页url
    names = []  # 房屋标签
    apartments = []  # 小区
    neighbourhoods = []  # 社区
    introductions = []  # 基本信息
    regions = []  # 行政区，详情页获取
    house_type = []  # 户型，拆分
    house_square = []  # 面积，拆分
    house_direction = []  # 朝向，拆分
    furniture = []  # 装修类型，拆分
    heated = []  # 房屋热度
    likes = []  # 关注人数，拆分
    times = []  # 发布时间，拆分
    prices = []  # 总价
    per_prices = []  # 单价


if __name__ == '__main__':
    target_pages = 100  # 爬取的页数总数
    file = open('file.csv', 'w+', newline='',encoding='utf-8')
    fieldnames = ['房屋标签', '小区', '社区', '行政区', '户型', '面积', '朝向', '装修类型', '关注人数', '发布时间', '总价', '平方米价格']  # 设置列名
    write = csv.DictWriter(file, fieldnames=fieldnames)  # 创建DictWriter对象
    write.writeheader()  # 写入表头
    for i in range(1, target_pages+1):
        clear_list()
        page_control(i)
        get_content(base_url, headers, data)
        get_region()
        split_info()
        split_heated()
        # print('第{}页'.format(i))
        # for j in range(0, 30):
        #     print(i,'\n')
        #     print({'房屋标签': names[j], '小区': apartments[j], '社区': neighbourhoods[j], '行政区': regions[j],
        #            '户型': house_type[j], '面积': house_square[j], '朝向': house_direction[j], '装修类型': furniture[j],
        #            '关注人数': likes[j], '发布时间': times[j], '总价': prices[j], '平方米价格': per_prices[j]})
        save_csv()
    print("爬取成功，保存为csv文件")
    file.close()
    # print(house_type)
