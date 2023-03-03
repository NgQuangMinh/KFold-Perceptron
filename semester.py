from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from sklearn.svm import SVC 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score , precision_score , recall_score , accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import KFold


data = pd.read_csv('./semester.csv')

cv = {'Female': 2, 'Male': 1 }
data['Gender'] = data['Gender'].map(cv)

cv = {'neutral or dissatisfied': 1, 'satisfied': 0 , '' : 0}
data['satisfaction'] = data['satisfaction'].map(cv)


print(data)


pla = Perceptron()


dt_Train , dt_Test = train_test_split(data , test_size=0.3 , shuffle=True)


k = 4

kf = KFold(n_splits=k, random_state=None)

mean_array = []
max_accuracy = 9999999999999999
i = 1
for(train_index, val_index) in kf.split(dt_Train):
	X_train , X_val = dt_Train.iloc[train_index , :-1] , dt_Train.iloc[val_index, :-1]
	y_train , y_val = dt_Train.iloc[train_index , -1] , dt_Train.iloc[val_index, -1]

	pla.fit(X_train, y_train)

	y_pred_val = pla.predict(X_val)
	y_pred_train = pla.predict(X_train)

	# mean_score = ( 1/accuracy_score( y_val, y_pred_val) + 1/ accuracy_score(y_train, y_pred_train) ) / 2

	error_train = 1 - accuracy_score(y_train, y_pred_train)
	error_val = 1 - accuracy_score(y_val, y_pred_val)

	sum_error = error_val + error_train 
	# TH1 = 90 + 50 => Sai nhiều ở tập val => Sai ít ở tập train => Đúng nhiều ở tập train + đúng ít ở tập val
	# TH2 = 70 + 70 => Sai đều nhau
	# KFOLD + LINEAR : ERROR : ĐỘ CHÊNH LỆCH GIÁ TRỊ THẬT VS GIÁ TRỊ DỰ ĐOÁN 
	# KFOLD + PLA : ERROR : TỔNG % DỰ ĐOÁN SAI CÀNG BÉ CÀNG TỐT => TỔNG % DỰ ĐOÁN LỚN CÀNG LỚN CÀNG TỐT
	
	# mean_score = 1/mean_score
	mean_array.append(sum_error)
	if( sum_error < max_accuracy):
		best_pla = pla.fit(X_train, y_train)
		max_accuracy = sum_error


# Đánh giá mô hình

y_pred_test = best_pla.predict(dt_Test.iloc[:,:-1])
y_test = dt_Test.iloc[:,-1]

accuracy_score = accuracy_score(y_test, y_pred_test)
precision_score = precision_score(y_test, y_pred_test)
recall_score = recall_score(y_test, y_pred_test)
f1_score = f1_score(y_test, y_pred_test)


form = Tk()
form.title("Dự đoán độ hài lòng của hành khách:")
form.geometry("600x700")

def showPredict():
	gender = checked_gender.get()
	age = tb_age.get()
	distance = textbox_distance.get()
	booking = textbox_online_booking.get()
	food = textbox_food.get()
	seat = textbox_seat.get()
	baggage = textbox_baggage.get()
	checkin = textbox_checkin.get()
	inflight = textbox_inflight.get()
	clean = textbox_clean.get()
	ddlay = textbox_ddelay.get()
	adelay = textbox_adelay.get()


	X_input = np.array([[gender , age , distance,booking , food , seat , baggage,checkin , inflight,clean,ddlay,adelay]], dtype=float)

	y_pred_input = best_pla.predict(X_input)
	if( y_pred_input[0] == 1):
		messagebox.showinfo("Kết quả dự đoán" , "Bạn đã bị ung thư [1]" )
	else:
		messagebox.showinfo("Kết quả dự đoán" , "Bạn hoàn toàn khỏe mạnh [0]" )


checked_gender = IntVar()


Label(form, text="Tuổi:").grid (column=1, row=2,  pady = (5, 5))
tb_age = Entry(form)
tb_age.grid(row = 2, column = 2, pady = 5)

Label(form, text="Giới tính:").grid (column=1, row=3 , pady = 5)
rad_gender1 = Radiobutton(form, text="Nam",variable=checked_gender, value=1)
rad_gender2 = Radiobutton(form, text="Nữ",variable=checked_gender, value=2)
rad_gender1.grid(column=2, row=3, )
rad_gender2.grid(column=3, row=3, )

lable_distance = Label(form, text="Khoảng cách :")
lable_distance.grid(row=4, column=1, padx=40, pady=5)
textbox_distance = Entry(form)
textbox_distance.grid(row=4, column=2)

lable_online_booking = Label(form, text="Điểm đặt vé online :")
lable_online_booking.grid(row=5, column=1, padx=40, pady=10)
textbox_online_booking = Entry(form)
textbox_online_booking.grid(row=5, column=2)

lable_food = Label(form, text="Điểm cho đồ ăn :")
lable_food.grid(row=6, column=1, padx=40, pady=5)
textbox_food = Entry(form)
textbox_food.grid(row=6, column=2)

lable_seat = Label(form, text="Độ thoải mái của chỗ ngồi :")
lable_seat.grid(row=7, column=1, padx=40, pady=5)
textbox_seat = Entry(form)
textbox_seat.grid(row=7, column=2)

lable_baggage = Label(form, text="Vận chuyển hành lý :")
lable_baggage.grid(row=8, column=1, padx=40, pady=5)
textbox_baggage = Entry(form)
textbox_baggage.grid(row=8, column=2)

lable_checkin = Label(form, text="Dịch vụ checkin :")
lable_checkin.grid(row=9, column=1, padx=40, pady=5)
textbox_checkin = Entry(form)
textbox_checkin.grid(row=9, column=2)

lable_inflight = Label(form, text="Dịch vụ trên máy bay :")
lable_inflight.grid(row=10, column=1, padx=40, pady=5)
textbox_inflight = Entry(form)
textbox_inflight.grid(row=10, column=2)

lable_clean = Label(form, text="Độ sạch sẽ:")
lable_clean.grid(row=11, column=1, padx=40, pady=5)
textbox_clean = Entry(form)
textbox_clean.grid(row=11, column=2)


lable_ddelay = Label(form, text="Delay khi cất cánh :")
lable_ddelay.grid(row=12, column=1, padx=40, pady=5)
textbox_ddelay = Entry(form)
textbox_ddelay.grid(row=12, column=2)

lable_adelay = Label(form, text="Delay khi hạ cánh :")
lable_adelay.grid(row=13, column=1, padx=40, pady=5)
textbox_adelay = Entry(form)
textbox_adelay.grid(row=13, column=2)


btn_predict = Button(form, text = 'Kết quả dự đoán theo PLA', command = showPredict)
btn_predict.grid(row = 14, column = 2, pady = 30, padx = 10)


label_acc = Label(form, text="Độ chính xác : " + str( round(accuracy_score,2)))
label_acc.grid(row=15 , column = 3 , pady = 10 , padx = 20)


label_f1 = Label(form, text="Độ đo F1 : " + str( round(f1_score,2)) )
label_f1.grid(row=16 , column = 3 , pady = 10 , padx = 20)


label_precision = Label(form, text="Độ đo precision : " + str(round( precision_score,2) ))
label_precision.grid(row=17 , column = 3 , pady = 10 , padx = 20)


label_recall = Label(form, text="Độ đo recall : " + str( round( recall_score,2)))
label_recall.grid(row=18 , column = 3 , pady = 10 , padx = 20)

label_k1 = Label(form, text="KFold lần 1 : " + str( round( mean_array[0],4) ))
label_k1.grid(row=15 , column = 1 , pady = 10 , padx = 20)

label_k2 = Label(form, text="KFold lần 2 : " + str( round( mean_array[1],4) ))
label_k2.grid(row=16 , column = 1 , pady = 10 , padx = 20)

label_k3 = Label(form, text="KFold lần 3 : " + str( round( mean_array[2],4) ))
label_k3.grid(row=17 , column = 1 , pady = 10 , padx = 20)

label_k4 = Label(form, text="KFold lần 4 : " + str( round( mean_array[3],4) ))
label_k4.grid(row=18 , column = 1 , pady = 10 , padx = 20)

form.mainloop()



