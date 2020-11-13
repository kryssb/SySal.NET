import struct

class FileInfo:
    Track = b'\x01'
    BaseTrack = b'\x02'
    Field = b'\x03'
    
class FileSection:
    Data = b'\x20'
    Header = b'\x40'
    
class FileFormat:
    Old = b'\x08\x00'
    Old2 = b'\x02\x00'
    NoExtents = b'\x01\x00'
    Normal = b'\x03\x00'
    NormalWithIndex = b'\x04\x00'
    NormalDoubleWithIndex = b'\x05\x00'
    Detailed = b'\x06\x00'
    MultiSection = b'\x07\x00' 
    
class Transform:
    def __init__(self):
        self.MXX = 1.0
        self.MXY = 0.0
        self.MYX = 0.0
        self.MYY = 1.0
        self.RX = 0.0
        self.RY = 0.0
        self.TX = 0.0
        self.TY = 0.0
        
class Vector2:
    def __init__(self):
        self.X = 0.0
        self.Y = 0.0
        
class Rectangle:
    def __init__(self):
        self.MinX = 0.0
        self.MinY = 0.0        
        self.MaxX = 0.0
        self.MaxY = 0.0

class View:
    def __init__(self, side = None, id = -1, px = 0.0, py = 0.0, topz = 0.0, bottomz = 0.0):
        self.Side = side
        self.Id = id
        self.TopZ = topz
        self.BottomZ = bottomz
        self.Position = Vector2()
        self.Position.X = px
        self.Position.Y = py
        self.Tracks = []
                
class Side:
    def __init__(self, topz = 0.0, bottomz = 0.0):
        self.Views = []
        self.Tracks = []
        self.TopZ = topz
        self.BottomZ = bottomz        

class TrackInfo:
    def __init__(self, field=None, grains=0, areasum=0, px=0.0, py=0.0, pz=0.0, sx=0.0, sy=0.0, sz=0.0, sigma=0.0, topz=0.0, bottomz=0.0, id=None, idfrag=None, idview=None, idtrack=None):
        self.Field = field
        self.Grains = grains
        self.AreaSum = areasum
        self.PX = px
        self.PY = py
        self.PZ = pz
        self.SX = sx
        self.SY = sy
        self.SZ = sz
        self.Sigma = sigma
        self.TopZ = topz
        self.BottomZ = bottomz
        self.Id = id
        self.IdFragment = idfrag
        self.IdView = idview
        self.IdTrack = idtrack
        
class BaseTrackInfo:
    def __init__(self, grains=0, areasum=0, px=0.0, py=0.0, pz=0.0, sx=0.0, sy=0.0, sz=0.0, sigma=0.0, topz=0.0, bottomz=0.0, toptrack=None, bottomtrack=None):
        self.Grains = grains
        self.AreaSum = areasum
        self.PX = px
        self.PY = py
        self.PZ = pz
        self.SX = sx
        self.SY = sy
        self.SZ = sz
        self.Sigma = sigma
        self.TopZ = topz
        self.BottomZ = bottomz
        self.TopTrack = toptrack
        self.BottomTrack = bottomtrack

class Zone:
    def __init__(self):
        self.Center = Vector2()
        self.Extents = Rectangle()
        self.Tracks = []
        self.Top = Side()
        self.Bottom = Side()
        self.Transform = Transform()

    def read(self, f):
        infotype = f.read(1)
        headerformat = f.read(2)
        if struct.unpack('<B', infotype)[0] != (struct.unpack('<B', FileInfo.Track)[0] | struct.unpack('<B', FileSection.Header)[0]):
                raise ValueError('Invalid file format (Track file with Header expected).')
        if headerformat != FileFormat.MultiSection \
            and headerformat != FileFormat.Detailed \
            and headerformat != FileFormat.NormalDoubleWithIndex \
            and headerformat != FileFormat.NormalWithIndex \
            and headerformat != FileFormat.Normal \
            and headerformat != FileFormat.NoExtents \
            and headerformat != FileFormat.Old \
            and headerformat != FileFormat.Old2:
                raise ValueError('Unknown format (unknown header).')                
        if (headerformat == FileFormat.MultiSection):
            if (f.read(1) != b'\x01'):
                raise ValueError('The first section in a TLG file must contain tracks!')
            f.read(8)
        if headerformat == FileFormat.Old:
            f.read(4)
        else:
            f.read(16) 
        if headerformat == FileFormat.Detailed or headerformat == FileFormat.MultiSection:
            self.Center.X = struct.unpack('<d',f.read(8))[0];
            self.Center.Y = struct.unpack('<d',f.read(8))[0];    
            self.Extents.MinX = struct.unpack('<d',f.read(8))[0];
            self.Extents.MaxX = struct.unpack('<d',f.read(8))[0];
            self.Extents.MinY = struct.unpack('<d',f.read(8))[0];
            self.Extents.MaxY = struct.unpack('<d',f.read(8))[0];
            self.Transform.MXX = struct.unpack('<d',f.read(8))[0];
            self.Transform.MXY = struct.unpack('<d',f.read(8))[0];
            self.Transform.MYX = struct.unpack('<d',f.read(8))[0];
            self.Transform.MYY = struct.unpack('<d',f.read(8))[0];
            self.Transform.TX = struct.unpack('<d',f.read(8))[0];
            self.Transform.TY = struct.unpack('<d',f.read(8))[0];
            self.Transform.RX = struct.unpack('<d',f.read(8))[0];
            self.Transform.RY = struct.unpack('<d',f.read(8))[0];
            topviews = [View() for x in range(0, struct.unpack('<I',f.read(4))[0])]
            bottomviews = [View() for x in range(0, struct.unpack('<I',f.read(4))[0])]
            self.Top = Side(topz=struct.unpack('<d',f.read(8))[0], bottomz=struct.unpack('<d',f.read(8))[0])
            self.Bottom = Side(topz=struct.unpack('<d',f.read(8))[0], bottomz=struct.unpack('<d',f.read(8))[0])
            self.Top.Views = topviews
            self.Bottom.Views = bottomviews
            for s in [self.Top, self.Bottom]:
                for i in range(0, len(s.Views)):
                    s.Views[i] = View(side=s, id=struct.unpack('<I',f.read(4))[0], px=struct.unpack('<d',f.read(8))[0], py=struct.unpack('<d',f.read(8))[0], topz=struct.unpack('<d',f.read(8))[0], bottomz=struct.unpack('<d',f.read(8))[0])    
            toptracks = [TrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            bottomtracks = [TrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            self.Tracks = [BaseTrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            for s in [(toptracks,topviews), (bottomtracks,bottomviews)]:
                for i in range(0, len(s[0])):
                    info = TrackInfo(field = struct.unpack('<I',f.read(4))[0], \
                        areasum = struct.unpack('<I',f.read(4))[0], \
                        grains = struct.unpack('<I',f.read(4))[0], \
                        px = struct.unpack('<d',f.read(8))[0], \
                        py = struct.unpack('<d',f.read(8))[0], \
                        pz = struct.unpack('<d',f.read(8))[0], \
                        sx = struct.unpack('<d',f.read(8))[0], \
                        sy = struct.unpack('<d',f.read(8))[0], \
                        sz = struct.unpack('<d',f.read(8))[0], \
                        sigma = struct.unpack('<d',f.read(8))[0], \
                        topz = struct.unpack('<d',f.read(8))[0], \
                        bottomz = struct.unpack('<d',f.read(8))[0])
                    s[0][i] = info
                    s[1][struct.unpack('<I',f.read(4))[0]].Tracks.append(info)
            self.Top.Tracks = toptracks
            self.Bottom.Tracks = bottomtracks
            for i in range(0, len(self.Tracks)):
                self.Tracks[i] = BaseTrackInfo(areasum = struct.unpack('<I',f.read(4))[0], \
                        grains = struct.unpack('<I',f.read(4))[0], \
                        px = struct.unpack('<d',f.read(8))[0], \
                        py = struct.unpack('<d',f.read(8))[0], \
                        pz = struct.unpack('<d',f.read(8))[0], \
                        sx = struct.unpack('<d',f.read(8))[0], \
                        sy = struct.unpack('<d',f.read(8))[0], \
                        sz = struct.unpack('<d',f.read(8))[0], \
                        sigma = struct.unpack('<d',f.read(8))[0], \
                        toptrack = self.Top.Tracks[struct.unpack('<I',f.read(4))[0]], \
                        bottomtrack = self.Bottom.Tracks[struct.unpack('<I',f.read(4))[0]])
                self.Tracks[i].TopZ = self.Tracks[i].TopTrack.TopZ
                self.Tracks[i].BottomZ = self.Tracks[i].BottomTrack.BottomZ   
            for s in [toptracks, bottomtracks]:
                for t in s:
                    t.IdFragment = struct.unpack('<I',f.read(4))[0]
                    t.IdView = struct.unpack('<I',f.read(4))[0]
                    t.IdTrack = struct.unpack('<I',f.read(4))[0] 
        elif headerformat == FileFormat.NormalDoubleWithIndex:
            self.Center.X = struct.unpack('<d',f.read(8))[0];
            self.Center.Y = struct.unpack('<d',f.read(8))[0];    
            self.Extents.MinX = struct.unpack('<d',f.read(8))[0];
            self.Extents.MaxX = struct.unpack('<d',f.read(8))[0];
            self.Extents.MinY = struct.unpack('<d',f.read(8))[0];
            self.Extents.MaxY = struct.unpack('<d',f.read(8))[0];
            f.read(4)
            topview = View()
            bottomview = View()
            toptracks = [TrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            bottomtracks = [TrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            self.Tracks = [BaseTrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            i = struct.unpack('<I',f.read(4))[0]
            topext = struct.unpack('<d',f.read(8))[0]
            topint = struct.unpack('<d',f.read(8))[0]
            bottomint = struct.unpack('<d',f.read(8))[0]
            bottomext = struct.unpack('<d',f.read(8))[0]
            f.read(2 * i)
            for s in [(toptracks,topview), (bottomtracks,bottomview)]:
                for i in range(0, len(s[0])):
                    info = TrackInfo(field = struct.unpack('<I',f.read(4))[0], \
                        areasum = struct.unpack('<I',f.read(4))[0], \
                        grains = struct.unpack('<I',f.read(4))[0], \
                        px = struct.unpack('<d',f.read(8))[0], \
                        py = struct.unpack('<d',f.read(8))[0], \
                        pz = struct.unpack('<d',f.read(8))[0], \
                        sx = struct.unpack('<d',f.read(8))[0], \
                        sy = struct.unpack('<d',f.read(8))[0], \
                        sz = struct.unpack('<d',f.read(8))[0], \
                        sigma = struct.unpack('<d',f.read(8))[0], \
                        topz = struct.unpack('<d',f.read(8))[0], \
                        bottomz = struct.unpack('<d',f.read(8))[0])
                    s[0][i] = info
                    s[1].Tracks.append(info)       
            self.Top = Side(topz=topext, bottomz=topint)
            self.Bottom = Side(topz=bottomint, bottomz=bottomext)
            self.Top.Views = [topview]
            self.Bottom.Views = [bottomview]
            topview.Side = self.Top
            bottomview.Side = self.Bottom
            topview.Tracks = toptracks
            bottomview.Tracks = bottomtracks
            topview.Id = 0
            bottomview.Id = 0
            topview.Position.X = self.Center.X
            topview.Position.Y = self.Center.Y
            bottomview.Position.X = self.Center.X
            bottomview.Position.Y = self.Center.Y
            topview.TopZ = self.Top.TopZ
            topview.BottomZ = self.Top.BottomZ
            bottomview.TopZ = self.Bottom.TopZ
            bottomview.BottomZ = self.Bottom.BottomZ    
            for i in range(0, len(self.Tracks)):
                self.Tracks[i] = BaseTrackInfo(areasum = struct.unpack('<I',f.read(4))[0], \
                        grains = struct.unpack('<I',f.read(4))[0], \
                        px = struct.unpack('<d',f.read(8))[0], \
                        py = struct.unpack('<d',f.read(8))[0], \
                        pz = struct.unpack('<d',f.read(8))[0], \
                        sx = struct.unpack('<d',f.read(8))[0], \
                        sy = struct.unpack('<d',f.read(8))[0], \
                        sz = struct.unpack('<d',f.read(8))[0], \
                        sigma = struct.unpack('<d',f.read(8))[0], \
                        toptrack = self.Top.Tracks[struct.unpack('<I',f.read(4))[0]], \
                        bottomtrack = self.Bottom.Tracks[struct.unpack('<I',f.read(4))[0]])
                self.Tracks[i].TopZ = self.Tracks[i].TopTrack.TopZ
                self.Tracks[i].BottomZ = self.Tracks[i].BottomTrack.BottomZ   
            for s in [toptracks, bottomtracks]:
                for t in s:
                    t.IdFragment = struct.unpack('<I',f.read(4))[0]
                    t.IdView = struct.unpack('<I',f.read(4))[0]
                    t.IdTrack = struct.unpack('<I',f.read(4))[0] 
        else:
            self.Center.X = struct.unpack('<f',f.read(4))[0];
            self.Center.Y = struct.unpack('<f',f.read(4))[0]; 
            if headerformat == FileFormat.Normal or headerformat == FileFormat.NormalWithIndex:
                self.Extents.MinX = struct.unpack('<f',f.read(4))[0];
                self.Extents.MaxX = struct.unpack('<f',f.read(4))[0];
                self.Extents.MinY = struct.unpack('<f',f.read(4))[0];
                self.Extents.MaxY = struct.unpack('<f',f.read(4))[0];
                f.read(4)
            else:
                f.read(20)
                self.Extents.MinX = self.Center.X
                self.Extents.MaxX = self.Center.X
                self.Extents.MinY = self.Center.Y
                self.Extents.MaxY = self.Center.Y
            topview = View()
            bottomview = View()
            toptracks = [TrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            bottomtracks = [TrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            self.Tracks = [BaseTrackInfo() for i in range(0, struct.unpack('<I',f.read(4))[0])]
            i = struct.unpack('<I',f.read(4))[0]
            topext = struct.unpack('<f',f.read(4))[0]
            topint = struct.unpack('<f',f.read(4))[0]
            bottomint = struct.unpack('<f',f.read(4))[0]
            bottomext = struct.unpack('<f',f.read(4))[0]
            f.read(2 * i)
            if headerformat == FileFormat.Normal or headerformat == FileFormat.NormalWithIndex:
                for s in [(toptracks,topview), (bottomtracks,bottomview)]:
                    for i in range(0, len(s[0])):
                        info = TrackInfo(field = struct.unpack('<I',f.read(4))[0], \
                            areasum = struct.unpack('<I',f.read(4))[0], \
                            grains = struct.unpack('<I',f.read(4))[0], \
                            px = struct.unpack('<f',f.read(4))[0], \
                            py = struct.unpack('<f',f.read(4))[0], \
                            pz = struct.unpack('<f',f.read(4))[0], \
                            sx = struct.unpack('<f',f.read(4))[0], \
                            sy = struct.unpack('<f',f.read(4))[0], \
                            sz = struct.unpack('<f',f.read(4))[0], \
                            sigma = struct.unpack('<f',f.read(4))[0], \
                            topz = struct.unpack('<f',f.read(4))[0], \
                            bottomz = struct.unpack('<f',f.read(4))[0])
                        s[0][i] = info
                        s[1].Tracks.append(info)       
            elif headerformat == FileFormat.NoExtents:
                for s in [(toptracks,topview), (bottomtracks,bottomview)]:
                    for i in range(0, len(s[0])):
                        info = TrackInfo(field = struct.unpack('<I',f.read(4))[0], \
                            areasum = struct.unpack('<I',f.read(4))[0], \
                            grains = struct.unpack('<I',f.read(4))[0], \
                            px = struct.unpack('<f',f.read(4))[0], \
                            py = struct.unpack('<f',f.read(4))[0], \
                            pz = struct.unpack('<f',f.read(4))[0], \
                            sx = struct.unpack('<f',f.read(4))[0], \
                            sy = struct.unpack('<f',f.read(4))[0], \
                            sz = struct.unpack('<f',f.read(4))[0], \
                            sigma = struct.unpack('<f',f.read(4))[0], \
                            topz = struct.unpack('<f',f.read(4))[0], \
                            bottomz = struct.unpack('<f',f.read(4))[0])
                        f.read(24)
                        s[0][i] = info
                        s[1].Tracks.append(info)       
            else:
                for s in [(toptracks,topview), (bottomtracks,bottomview)]:
                    for i in range(0, len(s[0])):
                        info = TrackInfo(field = struct.unpack('<I',f.read(4))[0], \
                            areasum = struct.unpack('<I',f.read(4))[0], \
                            grains = struct.unpack('<I',f.read(16)[0:4])[0], \
                            px = struct.unpack('<f',f.read(4))[0], \
                            py = struct.unpack('<f',f.read(4))[0], \
                            pz = struct.unpack('<f',f.read(4))[0], \
                            sx = struct.unpack('<f',f.read(4))[0], \
                            sy = struct.unpack('<f',f.read(4))[0], \
                            sz = struct.unpack('<f',f.read(4))[0], \
                            sigma = struct.unpack('<f',f.read(4))[0], \
                            topz = struct.unpack('<f',f.read(4))[0], \
                            bottomz = struct.unpack('<f',f.read(4))[0])
                        s[0][i] = info
                        s[1].Tracks.append(info)               
            self.Top = Side(topz=topext, bottomz=topint)
            self.Bottom = Side(topz=bottomint, bottomz=bottomext)
            self.Top.Views = [topview]
            self.Bottom.Views = [bottomview]
            topview.Side = self.Top
            bottomview.Side = self.Bottom
            topview.Tracks = toptracks
            bottomview.Tracks = bottomtracks
            topview.Id = 0
            bottomview.Id = 0
            topview.Position.X = self.Center.X
            topview.Position.Y = self.Center.Y
            bottomview.Position.X = self.Center.X
            bottomview.Position.Y = self.Center.Y
            topview.TopZ = self.Top.TopZ
            topview.BottomZ = self.Top.BottomZ
            bottomview.TopZ = self.Bottom.TopZ
            bottomview.BottomZ = self.Bottom.BottomZ
            if headerformat == FileFormat.Normal or headerformat == FileFormat.NormalWithIndex:
                for i in range(0, len(self.Tracks)):
                    self.Tracks[i] = BaseTrackInfo(areasum = struct.unpack('<I',f.read(4))[0], \
                            grains = struct.unpack('<I',f.read(4))[0], \
                            px = struct.unpack('<f',f.read(4))[0], \
                            py = struct.unpack('<f',f.read(4))[0], \
                            pz = struct.unpack('<f',f.read(4))[0], \
                            sx = struct.unpack('<f',f.read(4))[0], \
                            sy = struct.unpack('<f',f.read(4))[0], \
                            sz = struct.unpack('<f',f.read(4))[0], \
                            sigma = struct.unpack('<f',f.read(4))[0], \
                            toptrack = self.Top.Tracks[struct.unpack('<I',f.read(4))[0]], \
                            bottomtrack = self.Bottom.Tracks[struct.unpack('<I',f.read(4))[0]])
                    self.Tracks[i].TopZ = self.Tracks[i].TopTrack.TopZ
                    self.Tracks[i].BottomZ = self.Tracks[i].BottomTrack.BottomZ   
            elif headerformat == FileFormat.NoExtents:
                for i in range(0, len(self.Tracks)):
                    self.Tracks[i] = BaseTrackInfo(areasum = 0, \
                            grains = struct.unpack('<I',f.read(4))[0], \
                            px = struct.unpack('<f',f.read(4))[0], \
                            py = struct.unpack('<f',f.read(4))[0], \
                            pz = struct.unpack('<f',f.read(4))[0], \
                            sx = struct.unpack('<f',f.read(4))[0], \
                            sy = struct.unpack('<f',f.read(4))[0], \
                            sz = struct.unpack('<f',f.read(4))[0], \
                            sigma = struct.unpack('<f',f.read(28)[0:4])[0], \
                            toptrack = self.Top.Tracks[struct.unpack('<I',f.read(4))[0]], \
                            bottomtrack = self.Bottom.Tracks[struct.unpack('<I',f.read(4))[0]])
                    self.Tracks[i].TopZ = self.Tracks[i].TopTrack.TopZ
                    self.Tracks[i].BottomZ = self.Tracks[i].BottomTrack.BottomZ
            else:
                for i in range(0, len(self.Tracks)):
                    self.Tracks[i] = BaseTrackInfo(grains = struct.unpack('<I',f.read(4))[0], \
                            px = struct.unpack('<f',f.read(4))[0], \
                            py = struct.unpack('<f',f.read(4))[0], \
                            pz = struct.unpack('<f',f.read(4))[0], \
                            sx = struct.unpack('<f',f.read(4))[0], \
                            sy = struct.unpack('<f',f.read(4))[0], \
                            sz = struct.unpack('<f',f.read(4))[0], \
                            sigma = struct.unpack('<f',f.read(4))[0], \
                            toptrack = self.Top.Tracks[struct.unpack('<I',f.read(4))[0]], \
                            bottomtrack = self.Bottom.Tracks[struct.unpack('<I',f.read(4))[0]])
                    self.Tracks[i].areasum = toptrack.AreaSum + bottomtrack.AreaSum
                    self.Tracks[i].TopZ = self.Tracks[i].TopTrack.TopZ
                    self.Tracks[i].BottomZ = self.Tracks[i].BottomTrack.BottomZ        
            if headerformat == FileFormat.NormalWithIndex:
                for s in [toptracks, bottomtracks]:
                    for t in s:
                        t.IdFragment = struct.unpack('<I',f.read(4))[0]
                        t.IdView = struct.unpack('<I',f.read(4))[0]
                        t.IdTrack = struct.unpack('<I',f.read(4))[0] 