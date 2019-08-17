import numpy as np
data = open('kafka.txt', 'r').read().lower()
chars = list(set(data))

data_size, vocab_size = len(data), len(chars)


char_to_int = {ch:i for i, ch in enumerate(chars)}
int_to_char = {i:ch for i, ch in enumerate(chars)}


encoded_vector = np.zeros((vocab_size, 1))

hidden_size = 100
seq_length = 25
eta = 0.1
epochs = 1000000


#Initialize weights
Wxh = np.random.randn(hidden_size, vocab_size)
Whh = np.random.randn(hidden_size, hidden_size)
Why = np.random.randn(vocab_size, hidden_size)

#Initialize biases
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))
    
def LossFun(inputs, targets, hprev):
    x, h, y, p = dict(), dict(), dict(), dict()
    #Each of these are going to be seq_length long dicts
    h[-1] = np.copy(hprev)
    loss = 0
    
    #forward pass
    for t in range(len(inputs)):
        #Hot encode input character.
        x[t] = np.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        #Calculte h[t].
        h[t] = np.tanh(np.dot(Wxh, x[t]) + np.dot(Whh, h[t-1]) + bh)
        #calculate y.
        y[t] = np.dot(Why, h[t]) + by
        #calculate log probabilities.
        p[t] = np.exp(y[t])/np.sum(np.exp(y[t]))
        #calculate softmax cross entropy loss.
        loss += -np.log(p[t][targets[t], 0])


    #Backpropogaton through time
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    
    dhnext = np.zeros_like(h[0])
    #Gradient for next h
    for t in reversed(range(len(inputs))):
        #ouptut probabilities
        dy = np.copy(p[t])
        dy[targets[t]] -= 1
        
    
        #calculate dWhy
        dWhy += np.dot(dy, h[t].transpose())
        #calculate dby
        dby += dy
        
        #Calculate dbh
        dh = np.dot(Why.transpose(), dy) + dhnext
        dhraw = (1-h[t]*h[t])*dh
        dbh += dhraw
        #calculate dWxh  
        
        dWxh += np.dot(dhraw, x[t].transpose())
        
        #calculate dWhh
        dWhh += np.dot(dhraw, h[t-1].transpose())
        
        #calculate dhnext
        dhnext = np.dot(Whh.transpose(), dhraw)
    
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out = dparam)
        
    return (loss, dWxh, dWhh, dWhy, dbh, dby, h[len(inputs)-1])


def Sample(h, seed, n):
    #create hot encodeing for seed.
    x = np.zeros((vocab_size, 1))
    x[seed] = 1
    
    indexes = []
    #for the number of characters to be generated.
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        #calculate y.
        y = np.dot(Why, h) + by
        #calculate log probabilities.
        p = np.exp(y)/np.sum(np.exp(y))        
        #pick the letter with highest probability
        next_ = np.random.choice(range(vocab_size), p = p.ravel())
        #create hot encode vector
        x = np.zeros((vocab_size, 1))
        x[next_] = 1
        #append character index
        indexes.append(next_)

    text = ''.join(int_to_char[i] for i in indexes)
    print('.........................................TEXT....................................................')
    print(text)
    print()
    
#Main code to run loss function  and sample
pointer = 0
smooth_loss = -np.log(1.0/vocab_size)*seq_length
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
for epoch in range(epochs):
    if pointer + seq_length + 1 >= len(data) or epoch == 0:
        hprev = np.zeros((hidden_size, 1))
        pointer = 0
    
    inputs = [char_to_int[ch] for ch in data[pointer:pointer+seq_length]]
    targets = [char_to_int[ch] for ch in data[pointer+1:pointer+seq_length+1]]
    
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = LossFun(inputs, targets, hprev)
    
    smooth_loss = smooth_loss*0.999 + loss*0.001
    
    if epoch%1024 == 0:
        print('Iteration: ' + str(epoch) + '   Loss: ' + str(smooth_loss))
        Sample(hprev, inputs[0], 200)

    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                   [dWxh, dWhh, dWhy, dbh, dby],
                                   [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -eta*dparam/np.sqrt(mem + 1e-8)
    
    pointer += seq_length