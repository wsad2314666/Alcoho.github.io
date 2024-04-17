import pesq

#載入原始音訊和測試音訊
ref_audio='original.wav'
test_audio='test.wav'
score=pesq.pesq(ref_audio,test_audio,'wb')
print('PESQ Score:',score)