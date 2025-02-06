import socket
from struct import unpack 
import os

class CommunicatorCNS:
    def __init__(self, com_ip: str, com_port: int):
        """
        Initializes the CommunicatorCNS class with the specified IP and port.

        Args:
            com_ip (str): IP address for communication.
            com_port (int): Port number for communication.
        """
        # Create a UDP socket for receiving data
        self.resv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bind the socket to the provided IP and port
        self.resv_sock.bind((com_ip, com_port))
        # Buffer size for incoming data
        self.buffer_size = 46008
        
        # Initialize shared memory structure
        self.mem = self._make_mem_structure()

    def _make_mem_structure(self):
        """
        Creates a memory structure from a database file.

        Returns:
            dict: memory structure with PID as key and metadata as value.
        """

        # step 1. read para_list
        # para_list = ['KCNTOMS', 'NZON']
        dirpath = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dirpath, 'db.txt') 

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Database file not found: {file_path}")
        
        # step 2. build mem structure
        # mem = {para:{'Val':0, 'List':[]} for para in para_list}
        # db.txt 안에 있는 변수명과 변수타입만을 추출하려는 작업
        mem_structure = {}
        #memory structure 초기화
        with open(file_path, 'r') as f:
            for line in f:
                temp_ = line.strip().split('\t')
                if not temp_ or temp_[0] == '':
                    break  # End of file or empty line
                if temp_[0] != '#':
                    mem_structure[temp_[0]] = {
                        'Val': 0,
                        'List': []
                    }
        return mem_structure
    
    def read_data(self):
        # 데이터 읽기
        data, _ = self.resv_sock.recvfrom(self.buffer_size)
        data = data[8:]  # Skip the first 8 bytes
        
        for i in range(0, len(data), 20):
            pid, ival, sig, idx = unpack('12sihh', data[i:20 + i])
            pid, fval, sig, idx = unpack('12sfhh', data[i:20 + i])
            val = ival if sig == 0 else fval
            pid = pid.decode('utf-8').rstrip('\x00') # 공백제거

            if pid:
                self.mem[pid]['Val'] = val

        # read_data 이후 사용
        # 읽은 데이터를 저장을 할지 안할지 판단하기
        is_updated = False
        if len(self.mem['KCNTOMS']['List']) > 0:
            if self.mem['KCNTOMS']['Val'] != self.mem['KCNTOMS']['List'][-1]:
                for pid in self.mem.keys():
                    self.mem[pid]['List'].append(self.mem[pid]['Val'])
                is_updated = True
        else:
            for pid in self.mem.keys():
                self.mem[pid]['List'].append(self.mem[pid]['Val'])
            is_updated = True
        return is_updated      
        