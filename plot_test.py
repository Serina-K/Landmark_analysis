# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 18:50:39 2018

@author: serina
"""

import numpy as np
import matplotlib.pyplot as plt

# generate data

X = np.array([[0.5666666666666667, 0.24437950594556712],[0.06333333333333334, 0.26435252261797526],[1.0, 0],[0.17666666666666667, 0.26707887060183977],[0.043333333333333335, 0.27902741880459153],[0.41333333333333333, 0.26935233205270087],[1.0, 0],[0.3466666666666667, 0.2668092869133569],[0.08333333333333333, 0.2923333128762996],[0.06666666666666667, 0.312777237851993],[1.0, 0],[0.07, 0.26532353209647264],[0.73, 0.23913383167615168],[0.44, 0.2521577401105434],[0.1, 0.3001253653160764],[0.22, 0.25601675383149963],[0.023333333333333334, 0.3368821708402633],[0.0, 0.28002570777471225],[0.02666666666666667, 0.2933169590566406],[0.04666666666666667, 0.29574310344383126],[0.8266666666666667, 0.23952363463461535],[0.9133333333333333, 0.2377926195584501],[0.013333333333333334, 0.27653999596213974],[0.05333333333333334, 0.3095317106708717],[0.47333333333333333, 0.24281493126609502],[0.4666666666666667, 0.24991943145188505],[0.09, 0.28211899543057917],[0.03333333333333333, 0.2881204552765493],[0.99, 0],[0.19666666666666666, 0.2520935907637165],[0.04666666666666667, 0.2948563534647134],[0.97, 0.23993098549435685],[0.013333333333333334, 0.29181078432648294],[0.5166666666666667, 0.24432946669415856],[0.19333333333333333, 0.26106798887078386],[0.30333333333333334, 0.24207329906674363],[0.0, 0.2881399308746245],[0.8633333333333333, 0.2401218032679576],[0.03, 0.2904045014845705],[0.25666666666666665, 0.2608497269362586],[0.08333333333333333, 0.29426050367570944],[0.05333333333333334, 0.3085136490704701],[0.13666666666666666, 0.25407889368700265],[0.03333333333333333, 0.3125878652226693],[0.72, 0.2384654893230596],[0.13666666666666666, 0.27785383742358694],[0.013333333333333334, 0.2664566821701722],[0.013333333333333334, 0.3137628798306916],[0.056666666666666664, 0.2889480631525961],[0.01, 0.2902333844716739],[0.19333333333333333, 0.2517689974957084],[0.1, 0.26624158840054785],[0.5233333333333333, 0.24922525065707926],[0.013333333333333334, 0.29719638729767595],[0.9533333333333334, 0.23270834793654552],[0.04, 0.3193357774493841],[0.19333333333333333, 0.29259050104314355],[0.8166666666666667, 0.23663733462234043],[0.54, 0.24379952131230062],[0.6, 0.24930566432245985],[0.03, 0.2846284804750503],[0.08, 0.2750282693821303],[0.06, 0.2636909726873871],[1.0, 0],[0.043333333333333335, 0.31247033549101366],[1.0, 0],[1.0, 0],[0.06666666666666667, 0.27668988910968234],[0.016666666666666666, 0.28796665444929403],[0.65, 0.24719122369222865],[0.95, 0.2391598665240118],[0.11333333333333333, 0.2582501976962635],[0.41, 0.24143178976174345],[0.7366666666666667, 0.23851636302314091],[0.7866666666666666, 0.24188627782852817],[0.1, 0.2952983737331577],[0.84, 0.24297954855895657],[0.09666666666666666, 0.25994321537526566],[0.0, 0.31129661038599804],[0.02, 0.28292652397510965],[0.08333333333333333, 0.2923504754692912],[0.77, 0.24245792488653448],[0.03666666666666667, 0.2925541228842123],[0.3466666666666667, 0.2544517826055998],[0.11, 0.28546573928153846],[0.12333333333333334, 0.29406624077369137],[0.08333333333333333, 0.28114505626100394],[0.07333333333333333, 0.28020878518168074],[0.08333333333333333, 0.25723332273251187],[0.24333333333333335, 0.24787364455249036],[0.9066666666666666, 0.24818308951572407],[0.013333333333333334, 0.2853687998989918],[0.9033333333333333, 0.2405465441776128],[0.14666666666666667, 0.24963223440566285],[0.08666666666666667, 0.2999727760840322],[0.11, 0.2824147313905442],[0.6533333333333333, 0.24137447387631283],[0.05, 0.3123409987773097],[0.93, 0.23425500754570858],[0.0, 0.2808934106934971],[0.09, 0.27577996594272025],[0.05, 0.29315108817010593],[0.016666666666666666, 0.2544840910535182],[0.013333333333333334, 0.2770379949011008],[0.6666666666666666, 0.23917733180723938],[0.016666666666666666, 0.31829277347057233],[0.09, 0.25158642055482966],[0.9066666666666666, 0.2500946511009025],[0.3933333333333333, 0.2377436475044705],[0.23666666666666666, 0.2654543651911551],[0.15333333333333332, 0.26136223271138487],[1.0, 0],[0.44666666666666666, 0.2419994595252029],[0.6466666666666666, 0.2534751035392671],[0.07, 0.29703305798685503],[0.016666666666666666, 0.2813614362836335],[0.6733333333333333, 0.24518969301594826],[0.2866666666666667, 0.25108926829436506],[0.21666666666666667, 0.2731360805485302],[0.6133333333333333, 0.23981980606955375],[0.19666666666666666, 0.2556414051244933],[0.31333333333333335, 0.2547057899978859],[0.69, 0.2369202275669964],[0.0, 0.28012052403108084],[0.056666666666666664, 0.2807361608813959],[0.09666666666666666, 0.2844120212474354],[0.06, 0.2855522195620006],[0.7366666666666667, 0.23839133958423558],[0.03666666666666667, 0.2807124497491624],[0.3233333333333333, 0.25369307359319826],[0.52, 0.24888856334191592],[0.21666666666666667, 0.26678095229279075],[0.25333333333333335, 0.2632681214082627],[0.7533333333333333, 0.25449289470451814],[0.7966666666666666, 0.24202476395800396],[0.06666666666666667, 0.2613254704813226],[0.02, 0.2586717659241931],[0.02666666666666667, 0.281902480705442],[0.30666666666666664, 0.2591600689091575],[0.02, 0.27128046243378795],[0.013333333333333334, 0.28569626618282684],[0.10333333333333333, 0.2565767959840367],[0.57, 0.24556184582411036],[0.0, 0.2783978400024542],[0.6133333333333333, 0.23987171414792796],[0.89, 0.2372158735035956],[0.016666666666666666, 0.2595941514673156],[0.6933333333333334,0.25436071651342756],[0.3333333333333333, 0.2591478087667979],[0.02666666666666667, 0.29703465450444416],[0.25666666666666665, 0.258527325771313],[0.04666666666666667, 0.27626979201943297],[0.12333333333333334, 0.29853467093513053],[0.85, 0.2399121335798563],[0.19, 0.25664806531557605],[0.013333333333333334, 0.27740045107588507],[0.043333333333333335, 0.283308726934855],[0.15, 0.2652835723973704],[0.0, 0.3017179129440847],[0.20666666666666667, 0.28023588871883703],[0.03666666666666667, 0.2852427255021554],[0.08, 0.26909978654719696],[0.9833333333333333, 0.23426413316276426],[0.17333333333333334, 0.2729273347960785],[1.0, 0],[0.07333333333333333, 0.28276328087774033],[0.06333333333333334, 0.2817765005507403],[0.6566666666666666, 0.2403361556762903],[1.0, 0],[0.03, 0.2812243029903076],[0.21333333333333335, 0.29861843844739316],[0.0, 0.30172863897948005],[0.7, 0.24025673003233997],[0.1, 0.2825799815653237],[0.0, 0.28325033713034914],[0.12, 0.2673202403236423],[0.08, 0.2826001951857114],[0.99, 0.23215533180270861],[0.06666666666666667, 0.3082398782073362],[0.043333333333333335, 0.3126845131850568],[0.09666666666666666, 0.28075792996732307],[0.1, 0.3160747921641091],[0.9, 0.2387643653093793],[0.07666666666666666, 0.3040275980629529],[0.043333333333333335, 0.26548761133622384],[0.06, 0.30200249474196905],[0.2, 0.2874460481076691],[0.013333333333333334, 0.28638981624627824],[0.12666666666666668, 0.2589154845050764],[0.11333333333333333, 0.27383197897877726],[0.15666666666666668, 0.2612179789997741],[0.37333333333333335, 0.24787520129106608],[0.23666666666666666, 0.2526134607749469],[0.08666666666666667, 0.2704127476668291],[0.07, 0.30062469301216266],[0.21333333333333335, 0.26851089433104375],[0.21, 0.24849631771284492],[0.04666666666666667, 0.25430214051697697],[0.043333333333333335, 0.30355391913802865],[0.06333333333333334, 0.3228392140555788],[0.5733333333333334, 0.27419733719424466],[0.9733333333333334, 0.2352383255488096],[0.9666666666666667, 0.23567526802062672],[0.5333333333333333, 0.2511230870518439],[0.28, 0.24623850262433414],[0.8033333333333333, 0.2582877332671415],[0.03666666666666667, 0.2821429745582781],[0.9033333333333333, 0.2539448765570451],[0.7733333333333333, 0.25269209496930384],[0.11666666666666667, 0.261672542903836],[0.37333333333333335, 0.27750362988408694],[0.8966666666666666, 0.24113044198236797],[0.1, 0.25385809553657157],[0.2, 0.27049106503255277]])
y = np.array([5,3,5,2,1,4,3,1,2,2,5,2,3,3,2,3,2,3,4,3,5,4,1,2,3,3,2,3,4,2,2,4,2,3,4,4,2,5,3,4,3,1,3,3,4,3,2,3,3,4,3,3,4,3,5,3,3,4,3,4,2,2,3,5,2,4,5,2,2,2,4,1,5,4,5,1,4,2,1,2,2,5,2,3,2,4,4,3,3,5,5,1,5,3,3,3,4,3,3,2,2,2,4,2,4,3,3,5,5,3,2,5,4,2,2,1,2,3,3,4,2,2,4,2,3,4,2,3,2,3,2,4,4,2,5,3,3,1,3,3,2,2,3,3,4,5,2,2,3,3,3,2,2,5,2,3,2,4,3,3,3,2,5,3,5,2,3,5,5,1,2,1,4,2,2,2,3,4,2,3,3,2,5,2,3,3,3,2,2,3,3,4,4,2,3,3,3,3,3,2,5,5,5,4,4,3,2,3,5,2,3,5,3,4])




label1 = 1
label2 = 2
label3 = 3
label4 = 4
label5 = 5


        

fig = plt.figure()


ax = fig.add_subplot(1,1,1)



for i in range(len(X)):

    if y[i] == label1:
        color='red'
        label_name='label1'
    elif y[i] == label2:
        color='blue'
        label_name='label2'
    elif y[i] == label3:
        color='green'
        label_name='label3'
    elif y[i] == label4:
        color='yellow'
        label_name='label4'
    else:
        color='pink'
        label_name='label5'
    
    ax.scatter(X[i][0],X[i][1], c=color)
#    ax.scatter(X[i][1],X[i][0], c=color)


ax.set_title('plot')
ax.set_xlabel('Rb')
ax.set_ylabel('Aopen')


ax.grid(True)

#ax.legend(loc='upper left')
#plt.legend(bbox_to_anchor=(1, 1))


fig.show()

