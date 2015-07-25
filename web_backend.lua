require 'torch'
require 'nngraph'
require 'optim'
require 'lfs'
require 'nn'

require 'util.OneHot'
require 'util.misc'
JSON = (loadfile "util/JSON.lua")()


local redis = require 'redis'
local client = redis.connect('127.0.0.1', 6379)
local client2 = redis.connect('127.0.0.1', 6379)
local channels = {'cv_channel'}
local model_file = './onlie_model/model.t7'
local gpuid = 0
local seed = 123

-- check that cunn/cutorch are installed if user wants to use the GPU
if gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. gpuid .. '...')
        cutorch.setDevice(gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(seed)
    else
        print('Falling back on CPU mode')
        gpuid = -1 -- overwrite user setting
    end
end

if not lfs.attributes(model_file, 'mode') then
    print('Error: File ' .. model_file .. ' does not exist.')
end
checkpoint = torch.load(model_file)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- parse characters from a string
function get_char(str)
    local len  = #str
    local left = 0
    local arr  = {0, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc}
    local unordered = {}
    local start = 1
    local wordLen = 0
    while len ~= left do
        local tmp = string.byte(str, start)
        local i   = #arr
        while arr[i] do
            if tmp >= arr[i] then
                break
            end
            i = i - 1
        end
        wordLen = i + wordLen
        local tmpString = string.sub(str, start, wordLen)
        start = start + i
        left = left + i
		unordered[#unordered+1] = tmpString
    end
	return unordered
end

-- start listen
for msg in client:pubsub({subscribe = channels}) do
    if msg.kind == 'subscribe' then
        print('Subscribed to channel '..msg.channel)
    elseif msg.kind == 'message' then
        -- print('Received the following message from '..msg.channel.."\n  "..msg.payload.."\n")
        local req = JSON:decode(msg.payload)
        local primetext = '|' .. req['text'] .. '| '
        local session_id = req['sid']
        local seed = req['seed']
        local temperature = req['temp']

        -- initialize the rnn state to all zeros
        local current_state
        local num_layers = checkpoint.opt.num_layers
        current_state = {}
        for L = 1,checkpoint.opt.num_layers do
            -- c and h for all layers
            local h_init = torch.zeros(1, checkpoint.opt.rnn_size):float()
            if gpuid >= 0 then h_init = h_init:cuda() end
            table.insert(current_state, h_init:clone())
            table.insert(current_state, h_init:clone())
        end
        state_size = #current_state

        -- use input to init state
        torch.manualSeed(seed)
        for i,c in ipairs(get_char(primetext)) do
            prev_char = vocab[c]
            if prev_char then
                prev_char = torch.Tensor{vocab[c]}
                io.write(ivocab[prev_char[1]])
                if gpuid >= 0 then prev_char = prev_char:cuda() end
                local lst = protos.rnn:forward{prev_char, unpack(current_state)}
                -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
                current_state = {}
                for i=1,state_size do table.insert(current_state, lst[i]) end
                prediction = lst[#lst] -- last element holds the log probabilities
            end
        end
        -- start sampling/argmaxing
        result = ''
        not_end = true
        for i=1,1000 do
            -- log probabilities from the previous timestep
            -- make sure the output char is not UNKNOW
            real_char = 'UNKNOW'
            while(real_char == 'UNKNOW') do
                torch.manualSeed(seed+1)
                prediction:div(temperature) -- scale by temperature
                local probs = torch.exp(prediction):squeeze()
                probs:div(torch.sum(probs)) -- renormalize so probs sum to one
                prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
                real_char = ivocab[prev_char[1]]
            end

            -- forward the rnn for next character
            local lst = protos.rnn:forward{prev_char, unpack(current_state)}
            current_state = {}
            for i=1,state_size do table.insert(current_state, lst[i]) end
            prediction = lst[#lst] -- last element holds the log probabilities
            result = result .. ivocab[prev_char[1]]
            if string.find(result, '\n\n\n\n\n') then 
                not_end = false
                break 
            end
        end
        if not_end then result = result .. '……' end
        -- client2:set(session_id, result)
         client2:setex(session_id, 100, result)
    end
end
