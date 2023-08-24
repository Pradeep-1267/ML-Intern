from deposit_identifier import teeth_deposit_identifier

pred = teeth_deposit_identifier()

res = pred.predict("./images/1.jpeg" , th_1 = 0.6 , th_2 = 0.6 , display=True)
print(res)