# -------------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要: 获取指定文件夹下所有指定扩展名的文件路径
#           获取folder中所有子文件夹的路径
#   当前版本 : 0.0
#   作   者 ：于川汇 陈瑞侠
#   完成日期 : 2018-1-26
# -------------------------------------------------------------------
import os


def get_file_path(folder, exts=[], ret_full_path=False):
    '''
        作用:
            获取指定文件夹下所有指定扩展名的文件路径
        参数：
            folder       : 指定文件夹路径
            ret_full_path: 是否返回全路径，默认只返回符合条件的扩展名的文件名
            exts         : 扩展名列表
    '''
    if not (ret_full_path == True or ret_full_path == False):
        raise ValueError('输入参数只能是True或者False')
    if not (os.path.isdir(folder)):
        raise ValueError('输入参数必须是目录或者文件夹')
    if type(exts) == str:
        exts = [exts]
    result = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            (file_name, file_ext) = os.path.splitext(f)
            if (file_ext in exts) or (file_ext[1:] in exts) or (len(exts) == 0):
                if ret_full_path:
                    result.append(os.path.join(root, f))
                else:
                    result.append(f)
    return result


def get_subfolder_path(folder, ret_full_path=True):
    '''
        作用：
            获取folder中所有子文件夹的路径
        参数：
            ret_full_path: 是否返回全路径，默认返回子文件夹全路径
    '''
    if not (ret_full_path == True or ret_full_path == False):
        raise ValueError('输入参数只能是True或者False')
    if not (os.path.isdir(folder)):
        raise ValueError('输入参数必须是目录或者文件夹')
    result = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            if ret_full_path:
                result.append(os.path.join(root, d) + '/')
            else:
                result.append(d)
    return result


if __name__ == '__main__':
    # print(get_file_path('../../', 'jpg'))
    print(get_subfolder_path('../../'))
