require 'torch'
require 'nngraph'
require 'optim'
require 'lfs'
require 'nn'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('convert a gpu model to cpu one')
cmd:text()
cmd:text('Options')

cmd:argument('-load_model','model to convert')
cmd:argument('-save_file','the file path to save the converted model')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
local ok, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not ok then gprint('package cunn not found!') end
if not ok2 then gprint('package cutorch not found!') end
if ok and ok2 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
else
    print('No gpu found. Convert fail.')
    os.exit()
end

checkpoint = torch.load(opt.load_model)
checkpoint.protos.rnn:double()
checkpoint.protos.criterion:double()
torch.save(opt.save_file, checkpoint)
