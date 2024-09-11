import qrcode

# 输入文本
text = "你好，这是一段将要转换成QR码的文本"

# 生成QR码
qr = qrcode.QRCode(
    version=1,  # 控制 QR 码的大小，取值范围从 1 到 40
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # 控制错误纠正级别
    box_size=10,  # 控制每个框的像素大小
    border=4,  # 控制边框的大小
)
qr.add_data(text)
qr.make(fit=True)

# 生成图像
img = qr.make_image(fill='black', back_color='white')

# 保存图像
img.save("qrcode.png")

# 显示图像
img.show()
