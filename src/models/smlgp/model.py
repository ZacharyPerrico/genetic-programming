import numpy as np


class Linear:
    """
    Class representing a linear program during execution.
    """

    PC_INDEX = 0 # Index of the program counter in MEM0
    LINE_LENGTH = 4
    DEFAULT_VALUE_LIM = 256

    # Tuple of all operations and ordering to use if none is provided
    DEFAULT_OPS = (
        'STOP',
        'LOAD',
        'STORE',
        'ADD',
        'SUB',
        'MUL',
        'DIV',
        'IFEQ',
        # 'RAND',
        # 'DEL',
    )

    # Addressing modes
    # Does not include MEMx_DIRECT and MEMx_INDIRECT
    DEFAULT_ADDR_MODES = (
        'IMMEDIATE',
        'REGS_DIRECT',
        'REGS_INDIRECT',
        'CODE_DIRECT',
        'CODE_INDIRECT',
    )

    def __init__(self, mem, ops=DEFAULT_OPS, value_lim=DEFAULT_VALUE_LIM, **kwargs):
        self.value_lim = value_lim
        
        self.mem = mem.copy()
        self.regs = self.mem[0]
        self.code = self.mem[1]

        # Create a list and dict to map between op org strings and values
        self.valid_ops_list = ops
        self.valid_ops_dict = {v:u for u,v in enumerate(self.valid_ops_list)}

        # Create a list and dict to map between addr mode strings and values
        self.valid_addr_modes_list = list(Linear.DEFAULT_ADDR_MODES)
        self.valid_addr_modes_list += sum([[f'MEM{i}_DIRECT', f'MEM{i}_INDIRECT'] for i in range(2, len(self.mem))], [])
        self.valid_addr_modes_dict = {v:u for u,v in enumerate(self.valid_addr_modes_list)}

        # Mem can also be passed as containing strings
        # All instances of this must be replaced before execution
        valid_dict = self.valid_ops_dict | self.valid_addr_modes_dict
        for i, mem_bank in enumerate(mem):
            for j, value in enumerate(mem_bank):
                if type(value) == str:
                    self.mem[i][j] = valid_dict[value]


    def step(self):

        # Fetch the current line to be executed and unpack
        # Program counter loops back to start
        pc = int(self.regs[Linear.PC_INDEX])
        code_line = [self.code[(pc + i) % len(self.code)] for i in range(Linear.LINE_LENGTH)]
        op_code, target_reg, operand_spec, addr_mode = code_line

        # Values are modified to always be valid references
        op_code    = int(op_code)    % len(self.valid_ops_dict)
        target_reg = int(target_reg) % len(self.regs) if self.valid_ops_list[op_code] != 'RAND' else int(target_reg)
        addr_mode  = int(addr_mode)  % (len(self.mem) * 2 + 1)

        # Fetch the operand
        # Odd values are DIRECT
        # Nonzero even values are INDIRECT
        # Value of 0 is IMMEDIATE
        if addr_mode % 2 == 1:
            i = (addr_mode - 1) // 2
            j = int(operand_spec) % len(self.mem[i])
        elif addr_mode % 2 == 0 and addr_mode != 0:
            i = (addr_mode - 2) // 2
            j = self.regs[int(operand_spec) % len(self.regs)] % len(self.mem[i])
        else:
            i = 1
            j = (pc + 2) % len(self.mem[i])
        operand = self.mem[i][j]

        # Increment program counter
        self.regs[Linear.PC_INDEX] += Linear.LINE_LENGTH
        self.regs[Linear.PC_INDEX] %= self.value_lim

        # Perform the operation
        match self.valid_ops_list[op_code]:
            case 'STOP': return True
            case 'STORE': self.mem[i][j] = self.regs[target_reg] #% self.value_lim
            case 'LOAD': self.regs[target_reg] =                          operand  #% self.value_lim
            case 'ADD':  self.regs[target_reg] = (self.regs[target_reg] + operand) % self.value_lim
            case 'SUB':  self.regs[target_reg] = (self.regs[target_reg] - operand) % self.value_lim
            case 'MUL':  self.regs[target_reg] = (self.regs[target_reg] * operand) % self.value_lim
            case 'DIV':
                if operand != 0:
                    self.regs[target_reg] = (self.regs[target_reg] // operand) % self.value_lim
                else:
                    # self.regs[target_reg] = self.value_lim - 1
                    self.regs[target_reg] = 1
            case 'IFEQ':
                if self.regs[target_reg] != operand:
                    self.regs[Linear.PC_INDEX] = (self.regs[Linear.PC_INDEX] + Linear.LINE_LENGTH) % self.value_lim
            case 'RAND':
                low  = min(0, target_reg)
                high = max(0, target_reg)
                self.mem[i][j] = np.random.randint(low, high + 1) % self.value_lim
            case 'DEL': del self.mem[i][j]
            # case 'DUPE':
        return False


    def run(self, steps):
        """Run the machine for the given number of steps or until it reaches stop"""
        for _ in range(steps):
            if self.step():
                break
        return self


    def to_string(self, latex=False):
        SECTION_DELIM = ' & ' if latex else ' │ '
        STR_START = '\\mono{' if latex else "'"
        STR_END = '}' if latex else "'"
        VALUE_JOIN = ' & ' if latex else ', '
        LINE_END = '\\\\\n' if latex else '\n'
        string = ''
        for i, mem in enumerate(self.mem):
            string += f'self.mem[{i}]'
            if i == 0:
                string += ' (REGISTERS)'
            elif i == 1:
                string += ' (PROGRAM)'
            string += '\n'
            step_size = 1 if i == 0 else Linear.LINE_LENGTH
            for pc in range(0, len(mem), step_size):
                line = [mem[(pc + j) % len(mem)] for j in range(step_size)]
                # Line index
                string += f'  {pc:2}{SECTION_DELIM}'
                # Line values
                string += f'{VALUE_JOIN}'.join([f'{j:3}' for j in line])
                # Formated line as org
                if step_size != 1:
                    op_code, target_reg, operand_spec, addr_mode = line
                    op_code = int(op_code) % len(self.valid_ops_dict)
                    target_reg = int(target_reg) % len(self.regs) if self.valid_ops_list[op_code] != 'RAND' else int(target_reg)
                    addr_mode = int(addr_mode) % (len(self.mem) * 2 + 1)
                    # Replace number with constant reference
                    op_code = STR_START + self.valid_ops_list[op_code] + STR_END + VALUE_JOIN
                    addr_mode = self.valid_addr_modes_list[addr_mode] if addr_mode < len(
                        self.valid_addr_modes_list) else addr_mode
                    string += f'{SECTION_DELIM}{op_code:8}{target_reg:3}{VALUE_JOIN}{operand_spec:3}{VALUE_JOIN}{STR_START}{addr_mode}{STR_END}'
                string += LINE_END
        if latex:
            string.replace('_','\\_')
        return string

    def __str__(self):
        return self.to_string()


    def debug(self):
        """Enter debug mode"""
        print(self)
        while True:
            i = input('Steps: ')
            if i == '':
                self.step()
            else:
                self.run(int(i))
            print(self)


if __name__ == '__main__':
    pass

    REGS = [0,10,0,0]

    REGS[2] += REGS[1]
    REGS[3] += REGS[2]
    REGS[3] /= 3
    REGS[2] *= REGS[1]

    REGS[2] += REGS[1]
    REGS[3] += REGS[2]
    REGS[3] /= 3
    REGS[2] *= REGS[1]

    REGS[2] += REGS[1]
    REGS[3] += REGS[2]
    REGS[3] /= 3

    print(REGS)

    k = sum(i**2 for i in range(11))
    print(k)


    # c = [[
    #     0,6,0,0
    # ],[
    #     'ADD',    2,   1, 'REGS_DIRECT',
    #     'DIV',    2,   1, 'IMMEDIATE',
    #     'ADD',    3,   2, 'REGS_DIRECT',
    #     'DIV',    3,   3, 'IMMEDIATE',
    #     'MUL',    2,   1, 'REGS_DIRECT',
    # ]]
    #
    # l = Linear(c, value_lim=512, ops=('STOP', 'LOAD', 'ADD', 'SUB', 'MUL', 'DIV'))
    # l.debug()

    ## Mutate ##
    # org = [[
    #     0,
    #     0,
    # ],[
    #     Linear.RAND, 64, 1, Linear.VARS_DIRECT,
    #     Linear.RAND, 64, 1, Linear.MEM2_INDIRECT,
    # ],[
    #     0, 0, 0, 0,
    #     0, 0, 0, 0,
    # ]]

    ## Self-Rep / Crossover / Mutation ##
    # org = [[
    #     0, # PC
    #     0, # Random value
    #     0, # Copy pointer
    #     0, # Temp
    # ],[
    #     Linear.RAND,  1,  1, Linear.VARS_DIRECT,   # Generate random value
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
    #     Linear.LOAD,  3,  2, Linear.CODE_INDIRECT, # Load temp value from MEM2
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Execute next line if random value is 0
    #     Linear.STORE, 3,  2, Linear.MEM2_INDIRECT, # Store temp value into MEM2
    #     Linear.ADD,   2,  1, Linear.IMMEDIATE,     # Increment copy pointer
    # ],[
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    #     32, 32, 32, 32,
    # ]]

    ## Multiply ##
    # code = [[
    #     0,
    #     3,
    #     5,
    #     0,
    # ],[
    #     'IFEQ', 2,  4, 'CODE_DIRECT',
    #     'STOP', 2,  9, 'REGS_DIRECT',
    #     'SUB',  2, 15, 'CODE_DIRECT',
    #     'ADD',  3, 13, 'REGS_DIRECT',
    # ]]

    ## Evolved Self Rep ##
    # org = [
    #     [Linear.SUB   ,  1 ,  3 , Linear.IMMEDIATE],
    #     [Linear.LOAD  ,  2 ,  4 , Linear.MEM_INDIRECT],
    #     [Linear.STORE ,  2 , 10 , Linear.OUT_INDIRECT],
    #     # [Linear.IFEQ  ,  1 ,  8 , Linear.MEM_INDIRECT],
    #     [Linear.SUB   ,  1 ,  3 , Linear.OUT_DIRECT],
    # ]

    ## One Point Crossover ##
    # org = [[
    #     0, # PC
    #     0, # Copy pointer
    #     0, # Temp
    # ],[
    #     Linear.IFEQ,  1,  0, Linear.IMMEDIATE,     # Check if copy pointer is 0
    #     Linear.RAND,4*8,  1, Linear.VARS_DIRECT,   # Randomly move the copy pointer
    #     Linear.LOAD,  2,  1, Linear.MEM2_INDIRECT, # Load temp value from MEM2
    #     Linear.STORE, 2,  1, Linear.MEM3_INDIRECT, # Store temp value into MEM3
    #     Linear.ADD,   1,  1, Linear.IMMEDIATE,     # Increment copy pointer
    #     Linear.IFEQ,  1,4*8, Linear.IMMEDIATE,     # Check if copy pointer is at last position
    #     Linear.STOP,  0,  0, Linear.IMMEDIATE,     # End execution
    # ],
    #     [1] * 32,
    #     [2] * 32,
    # ]

    # org = [[
    #
    # ],[
    #     'LOAD', 3, 19, 'CODE_INDIRECT',
    #     'ADD', 3, 27, 'REGS_INDIRECT',
    #     'ADD', 0, 17, 'CODE_INDIRECT',
    #     'ADD', 1, 12, 'REGS_INDIRECT',
    # ]]

    # x = sum([
    #     15, 10, 11, 14,
    #     13, 13, 6, 6,
    #     3, 5, 2, 11,
    #     2, 5, 6, 6,
    # ])
    # print(x)

    # l = Linear(code, ops=('STOP', 'LOAD', 'STORE', 'ADD', 'SUB', 'IFEQ'), value_lim=32)
    # print(l)
    # l.run(100)
    # print(l)
    # l.run(4000)
    # print(l)

