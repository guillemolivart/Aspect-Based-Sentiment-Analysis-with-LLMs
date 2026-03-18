import sys
import json

#------------------------------------
def flatten(d, pref="") :
   '''
   flatten a dictionary into a list of keypath:value
   '''
   if type(d) in [str, int, float, bool]:
      return pref+":"+str(d)
   elif type(d) == list :
      return [flatten(x, pref) for x in d]
   elif type(d) == dict :
      f = []
      for k in d :
         ls = flatten(d[k], pref+f".{k}")
         if type(ls)==list : f.extend(ls)
         else: f.append(ls)
      return f
   else :
      print("ERROR - Cannot flaten '{type(d)}'")
      return None   

#------------------------------------
def counts(pred, gold):
   ''' 
   Given a prediction and the expected gold result, 
   count number of predicted fields, 
   number of expected, and number of matches
   '''
   pred = flatten(pred)
   gold= flatten(gold)
   prd = len(pred)  # num. of predicted 
   exp = len(gold)  # num of expected
   ok = 0 # num of matches
   for k in pred :
      if k in gold: 
          ok += 1

   return prd, exp, ok

#------------------------------------
def stats(prd, exp, ok) :
   ''' 
   Given the count of predicted, expected and matches, 
   compute precisio, recall and F1
   '''
   P = 100.0*ok/prd if prd>0 else 0
   R = 100.0*ok/exp if exp>0 else 0
   F = 2*P*R/(P+R) if P+R>0 else 0
   return P, R, F


#------------------------------------
with open(sys.argv[1]) as jf :
   examples = json.load(jf)
   
prd_t, exp_t, ok_t = 0, 0, 0
PM, RM, FM = 0, 0, 0
for ex in examples :

   # get current example stats
   prd, exp, ok = counts(ex["prediction"], ex["gold"])

   # accumulate total counts for micro P/R/F1 later
   prd_t += prd
   exp_t += exp
   ok_t += ok

   # compute P/R/F1 for current example, and print stats
   P, R, F = stats(prd,exp,ok)
   print(f"{ex['id']:35s}\t   {P:5.1f} {R:5.1f} {F:5.1f}")
   
   # acumulate sums for for macro P/R/F1 average later
   PM += P
   RM += R
   FM += F
   

# macro P/R/F1: average of P/R/F1 over all examples
PM /= len(examples)
RM /= len(examples)
FM /= len(examples)

# micro P/R/F1: P/R/F1 over all fields of all examples
Pm, Rm, Fm = stats(prd_t, exp_t, ok_t)

# print averages
print("         P     R     F")
print(f"M.avg  {PM:5.1f} {RM:5.1f} {FM:5.1f}")
print(f"m.avg  {Pm:5.1f} {Rm:5.1f} {Fm:5.1f}")

