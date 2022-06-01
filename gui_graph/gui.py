#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tkinter import *  # 导入 Tkinter 库
import tkinter.filedialog
import os
from tkinter import messagebox
from draw_graph import draw



def selectPath():
    path_ = tkinter.filedialog.askopenfilename()
    name = os.path.split(path_)
    index = len(name)
    Vedio_name.set(name[index-1])



def predict():
  result = draw(Vedio_name.get())
  """
  if result == None:
      messagebox.showinfo('提示', '该视频不存在！')
  """

if __name__ == '__main__':
    root = Tk()  # 创建窗口对象的背景色
    root.title("短视频人物性格预测")  # 窗口标题
    root.geometry("500x350")  # 窗口大小

    # welcome image
    canvas = Canvas(root, height=300, width=500)
    image_file = PhotoImage(file='welcome.gif')
    image = canvas.create_image(30, 20, anchor='nw', image=image_file)
    canvas.pack(side='top')


    Label(root, text='请输入短视频的名称：').place(x=185, y=180)
    Vedio_name = StringVar()
    Entry(root, textvariable=Vedio_name).place(x=155, y=210)
    Button(root, text="选择", command=selectPath).place(x=350, y=207)
    Button(root, text='预测', command=predict).place(x=190, y=240)
    Button(root, text='退出', command=quit).place(x=255, y=240)
    # 循环
    root.mainloop()
