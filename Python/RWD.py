import struct

class Track:
    def __init__(self):
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.SX = 0.0
        self.SY = 0.0
        self.Grains = 0
        self.AreaSum = 0
        self.Sigma = 0.0
        self.TopZ = 0.0
        self.BottomZ = 0.0

    def project(self, zp):
        return [self.X + self.SX * (zp - self.Z), self.Y + self.SY * (zp - self.Z), zp]
        

class LayerInfo:
    def __init__(self):
        self.Z = 0
        self.Grains = 0


class Side:
    def __init__(self):
        self.Layers = []
        self.Tracks = []
        self.TopZ = 0.0
        self.BottomZ = 0.0
        self.PosX = 0.0
        self.PosY = 0.0
        self.MapPosX = 0.0
        self.MapPosY = 0.0
        self.MXX = 1.0
        self.MXY = 0.0
        self.MYX = 0.0
        self.MYY = 1.0
        self.IMXX = 1.0
        self.IMXY = 0.0
        self.IMYX = 0.0
        self.IMYY = 0.0

    def invert(self):
        det = 1.0 / (self.MXX * self.MYY - self.MXY * self.MYX)
        self.IMXX = self.MYY * det
        self.IMXY = -self.MXY * det
        self.IMYX = -self.MYX * det
        self.IMYY = self.MXX * det

    def transform(self, tk):
        tk1 = Track()
        tk1.AreaSum = tk.AreaSum
        tk1.Grains = tk.Grains
        tk1.Sigma = tk.Sigma
        tk1.TopZ = tk.TopZ
        tk1.BottomZ = tk.BottomZ
        tk1.SX = self.MXX * tk.SX + self.MXY * tk.SY
        tk1.SY = self.MYX * tk.SX + self.MYY * tk.SY
        tk1.Z = tk.Z
        tk1.X = self.MXX * tk.X + self.MXY * tk.Y + self.MapPosX
        tk1.Y = self.MYX * tk.X + self.MYY * tk.Y + self.MapPosY
        return tk1


class View:
    def __init__(self):
        self.TileX = -1
        self.TileY = -1
        self.Top = Side()
        self.Bottom = Side()
        
        
class Fragment:
    def __init__(self):
        self.Id = [0, 0, 0, 0]
        self.Index = 0
        self.StartView = 0
        self.Views = []
        
    def read(self, f):
        if f.read(1) != b'\x66':
            raise ValueError('Invalid file format.')
        if f.read(2) != b'\x03\x07':
            raise ValueError('Invalid header.')
        self.Id = [struct.unpack('<I',f.read(4))[0] for i in range(0,4)]
        self.Index = struct.unpack('<I',f.read(4))[0]
        self.StartView = struct.unpack('<I',f.read(4))[0]
        views = struct.unpack('<I',f.read(4))[0]
        fitcorrectiondatasize = struct.unpack('<I',f.read(4))[0]
        codingmode = f.read(4)
        if codingmode != b'\x02\x01\x00\x00':
            raise ValueError('Invalid coding mode (only GrainSuppression supported).')
        f.read(256)
        for vi in range(0, views):
            v = View()
            v.TileX = struct.unpack('<I',f.read(4))[0]
            v.TileY = struct.unpack('<I',f.read(4))[0]
            v.Top.PosX = struct.unpack('<d',f.read(8))[0]
            v.Bottom.PosX = struct.unpack('<d',f.read(8))[0]
            v.Top.PosY = struct.unpack('<d',f.read(8))[0]
            v.Bottom.PosY = struct.unpack('<d',f.read(8))[0]
            v.Top.MapPosX = struct.unpack('<d',f.read(8))[0]
            v.Bottom.MapPosX = struct.unpack('<d',f.read(8))[0]
            v.Top.MapPosY = struct.unpack('<d',f.read(8))[0]
            v.Bottom.MapPosY = struct.unpack('<d',f.read(8))[0]
            v.Top.MXX = struct.unpack('<d',f.read(8))[0]
            v.Top.MXY = struct.unpack('<d',f.read(8))[0]
            v.Top.MYX = struct.unpack('<d',f.read(8))[0]
            v.Top.MYY = struct.unpack('<d',f.read(8))[0]
            v.Top.invert()
            v.Bottom.MXX = struct.unpack('<d',f.read(8))[0]
            v.Bottom.MXY = struct.unpack('<d',f.read(8))[0]
            v.Bottom.MYX = struct.unpack('<d',f.read(8))[0]
            v.Bottom.MYY = struct.unpack('<d',f.read(8))[0]
            v.Bottom.invert()
            layers = struct.unpack('<I',f.read(4))[0]
            v.Top.Layers = [LayerInfo() for i in range(0, layers)]
            layers = struct.unpack('<I',f.read(4))[0]
            v.Bottom.Layers = [LayerInfo() for i in range(0, layers)]
            v.Top.TopZ = struct.unpack('<d',f.read(8))[0]
            v.Top.BottomZ = struct.unpack('<d',f.read(8))[0]
            v.Bottom.TopZ = struct.unpack('<d',f.read(8))[0]
            v.Bottom.BottomZ = struct.unpack('<d',f.read(8))[0]
            f.read(1)
            f.read(1)
            tracks = struct.unpack('<I',f.read(4))[0]
            v.Top.Tracks = [Track() for i in range(0, tracks)]
            tracks = struct.unpack('<I',f.read(4))[0]
            v.Bottom.Tracks = [Track() for i in range(0, tracks)]
            self.Views.append(v)
        for v in self.Views:
            for s in [v.Top, v.Bottom]:
                for ly in s.Layers:
                    ly.Grains = struct.unpack('<I',f.read(4))[0]
                    ly.Z = struct.unpack('<d',f.read(8))[0]
        for v in self.Views:
            for s in [v.Top, v.Bottom]:
                for t in s.Tracks:
                    t.AreaSum = struct.unpack('<I',f.read(4))[0]
                    t.Grains = struct.unpack('<I',f.read(4))[0]
                    t.X = struct.unpack('<d',f.read(8))[0]
                    t.Y = struct.unpack('<d',f.read(8))[0]
                    t.Z = struct.unpack('<d',f.read(8))[0]
                    t.SX = struct.unpack('<d',f.read(8))[0]
                    t.SY = struct.unpack('<d',f.read(8))[0]
                    f.read(8)
                    t.Sigma = struct.unpack('<d',f.read(8))[0]
                    t.TopZ = struct.unpack('<d',f.read(8))[0]
                    t.BottomZ = struct.unpack('<d',f.read(8))[0]
