using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.TotalScan.Flexi
{
    /// <summary>
    /// Specifies the origin of a data element such as microtrack/volume track/vertex.
    /// </summary>
    public class DataSet
    {
        /// <summary>
        /// The type of data.
        /// </summary>
        public string DataType;
        /// <summary>
        /// The id of the data.
        /// </summary>
        public long DataId;
        /// <summary>
        /// Checks whether two datasets are equal.
        /// </summary>
        /// <param name="a">the first dataset to be compared.</param>
        /// <param name="b">the second dataset to be compared.</param>
        /// <returns><c>true</c> if the two are found equal, <c>false</c> otherwise.</returns>
        public static bool AreEqual(DataSet a, DataSet b) { return a.DataId == b.DataId && (String.Compare(a.DataType, b.DataType, true) == 0); }
        /// <summary>
        /// Checks whether two dataset have the same datatype.
        /// </summary>
        /// <param name="a">the first dataset to be compared.</param>
        /// <param name="b">the second dataset to be compared.</param>
        /// <returns><c>true</c> if the two datasets are found equal, <c>false</c> otherwise.</returns>
        public static bool AreSameType(DataSet a, DataSet b) { return (String.Compare(a.DataType, b.DataType, true) == 0); }
        /// <summary>
        /// Generates a text dump of the object.
        /// </summary>
        /// <returns>a string with a text dump of the object.</returns>
        public override string ToString()
        {
            return DataType + " " + DataId.ToString("G20", System.Globalization.CultureInfo.InvariantCulture);
        }
    }

    /// <summary>
    /// A segment in a FlexiVolume.
    /// </summary>
    public class Segment : SySal.TotalScan.Segment, ICloneable
    {
        /// <summary>
        /// Builds a text representation of a segment.
        /// </summary>
        /// <returns>a string with the dump of the information in the segment.</returns>
        public override string ToString()
        {
            string ds = "";
            try
            {
                ds = DataSet.ToString();
            }
            catch (Exception)
            {
                ds = "-UNKNOWN-";
            }
            return base.ToString() + "\r\nDataSet: " + ds;
        }
        /// <summary>
        /// Makes a copy of a segment, providing minimum functions of a SySal.TotalScan.Flexi.Segment.
        /// </summary>
        /// <param name="s">the segment to be copied.</param>
        /// <param name="ds">the dataset to assign the segment to.</param>
        /// <returns>the copy of the segment.</returns>
        public static SySal.TotalScan.Flexi.Segment Copy(SySal.TotalScan.Segment s, SySal.TotalScan.Flexi.DataSet ds)
        {
            if (s is SySal.TotalScan.Flexi.Segment)
            {
                SySal.TotalScan.Flexi.Segment ns = (s as SySal.TotalScan.Flexi.Segment).Clone() as SySal.TotalScan.Flexi.Segment;
                ns.DataSet = ds;
                return ns;
            }
            return new SySal.TotalScan.Flexi.Segment(s, ds);
        }
        /// <summary>
        /// Property backer member for <c>DataSet</c>.
        /// </summary>
        protected DataSet m_DataSet;
        /// <summary>
        /// The data set this segment belongs to.
        /// </summary>
        public virtual DataSet DataSet
        {
            get { return m_DataSet; }
            set { m_DataSet = value; }
        }
        /// <summary>
        /// Builds a new Segment from a TotalScan Segment.
        /// </summary>
        /// <param name="seg">the segment to copy.</param>
        /// <param name="ds">the dataset marker to assign this data element to.</param>
        /// <remarks>The information of the original layer and track are lost.</remarks>
        public Segment(SySal.TotalScan.Segment seg, DataSet ds)
        {
            m_Index = seg.Index;
            m_Info = seg.Info;
            m_DataSet = ds;
        }
        /// <summary>
        /// Sets the layer and position within the layer for this segment.
        /// </summary>
        /// <param name="ly">the layer to attach the segment to.</param>
        /// <param name="lypos">the position of the segment in the layer.</param>
        /// <remarks>The layer receives no notification of the newly attached segment. External code must maintain the consistency.</remarks>
        public virtual void SetLayer(SySal.TotalScan.Layer ly, int lypos)
        {
            m_LayerOwner = ly;
            m_PosInLayer = lypos;
        }
        /// <summary>
        /// Sets the owner track and position within the track for this segment.
        /// </summary>
        /// <param name="ly">the track to attach the segment to.</param>
        /// <param name="lypos">the position of the segment in the track.</param>
        /// <remarks>The track receives no notification of the newly attached segment. External code must maintain the consistency.</remarks>
        public virtual void SetTrack(SySal.TotalScan.Track tk, int tkpos)
        {
            m_TrackOwner = tk;
            m_PosInTrack = tkpos;
        }
        /// <summary>
        /// Sets the geometrical parameters for the segment.
        /// </summary>
        /// <param name="info">the new geometrical parameters.</param>
        /// <remarks>The owner track (if any) receives no notification of the change. External code must maintain the consistency.</remarks>
        public virtual void SetInfo(SySal.Tracking.MIPEmulsionTrackInfo info)
        {
            m_Info = info;
        }
        /// <summary>
        /// Sets the index for the segment.
        /// </summary>
        /// <param name="ix">the new index to be set.</param>
        public void SetIndex(SySal.TotalScan.Index ix)
        {
            m_Index = ix;
        }

        #region ICloneable Members

        public virtual object Clone()
        {
            return new Segment(this, this.DataSet);
        }

        #endregion
    }

    /// <summary>
    /// A layer in a FlexiVolume.
    /// </summary>
    public class Layer : SySal.TotalScan.Layer
    {
        /// <summary>
        /// Sets the ordering number of the layer in the FlexiVolume.
        /// </summary>
        /// <param name="id">the new ordering number.</param>
        public virtual void SetId(int id)
        {
            m_Id = id;
        }
        /// <summary>
        /// Assigns a brick identifier to the layer. It is useful for old volumes that did not come with brick assignment.
        /// </summary>
        /// <param name="bkid">the brick to which the layer belongs.</param>
        public virtual void SetBrickId(long bkid)
        {
            m_BrickId = bkid;
        }
        /// <summary>
        /// Copies an existing layer into a FlexiLayer.
        /// </summary>
        /// <param name="ly">the original layer.</param>
        /// <param name="ds">the dataset to which the segments of this layer should be attached.</param>
        /// <remarks>The segments in the original dataset are copied to the new one, setting the LayerOwner and the position in the layer to the newly created object.</remarks>
        public Layer(SySal.TotalScan.Layer ly, SySal.TotalScan.Flexi.DataSet ds) : base()
        {
            m_Id = ly.Id;
            m_RefCenter = ly.RefCenter;
            m_BrickId = ly.BrickId;
            m_SheetId = ly.SheetId;
            m_Side = ly.Side;
            m_DownstreamZ = ly.DownstreamZ;
            m_UpstreamZ = ly.UpstreamZ;
            m_DownstreamZ_Updated = true;
            m_UpstreamZ_Updated = true;            
            SetAlignmentData(ly.AlignData);
            m_RadiationLength = ly.RadiationLengh;
            m_UpstreamRadiationLength = ly.UpstreamRadiationLength;
            m_DownstreamRadiationLength = ly.DownstreamRadiationLength;
            int n = ly.Length;
            int i;
            Segments = new SySal.TotalScan.Segment[n];
            for (i = 0; i < n; i++)
            {
                Segment seg = SySal.TotalScan.Flexi.Segment.Copy(ly[i], ds);
                seg.SetLayer(this, i);
                Segments[i] = seg;
            }            
        }
        /// <summary>
        /// Creates an empty layer.
        /// </summary>        
        /// <param name="id">the order number in the FlexiVolume.</param>
        /// <param name="brickid">the brick id.</param>
        /// <param name="sheet">the sheet in the brick.</param>
        /// <param name="side">the side (0 for base tracks, 1 for downstream, 2 for upstream).</param>        
        public Layer(int id, long brickid, int sheet, short side) : base()
        {
            m_BrickId = brickid;
            m_SheetId = sheet;
            m_Side = side;
            m_Id = id;
            m_DownstreamZ_Updated = false;
            m_UpstreamZ_Updated = false;
            AlignmentData al = new AlignmentData();
            al.AffineMatrixXX = al.AffineMatrixYY = 1.0;
            al.AffineMatrixXY = al.AffineMatrixYX = 0.0;
            al.TranslationX = al.TranslationY = al.TranslationZ = 0.0;
            al.DShrinkX = al.DShrinkY = 1.0;
            al.SAlignDSlopeX = al.SAlignDSlopeY = 0.0;
            SetAlignmentData(al);
            Segments = new Segment[0];
        }
        /// <summary>
        /// Sets the upstream Z for the layer.
        /// </summary>
        /// <param name="z">the new Z value.</param>
        /// <remarks>This method does not move segment Zs.</remarks>
        public virtual void SetUpstreamZ(double z)
        {
            m_UpstreamZ = z;
            m_UpstreamZ_Updated = true;
        }
        /// <summary>
        /// Sets the downstream Z for the layer.
        /// </summary>
        /// <param name="z">the new Z value.</param>
        /// <remarks>This method does not move segment Zs.</remarks>
        public virtual void SetDownstreamZ(double z)
        {
            m_DownstreamZ = z;
            m_DownstreamZ_Updated = true;
        }
        /// <summary>
        /// Sets the reference center of the layer.
        /// </summary>
        /// <param name="r">the new reference center.</param>
        /// <remarks>This method does not move segment Zs.</remarks>
        public virtual void SetRefCenter(SySal.BasicTypes.Vector r)
        {
            m_RefCenter = r;
        }
        /// <summary>
        /// Sets the Z references of the layers, also adjusting segment positions.
        /// </summary>
        /// <param name="newupz">the new value of the upstream Z.</param>
        /// <param name="newdownz">the new value of the downstream Z.</param>
        /// <remarks>The new value of the Z of the reference center is computed by an affine transformation, as well as those of segments.</remarks>
        public virtual void DisplaceAndClampZ(double newupz, double newdownz)
        {
            double idz = (newdownz - newupz) / (m_DownstreamZ - m_UpstreamZ);            
            int i;
            for (i = 0; i < Segments.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = Segments[i].Info;
                info.Intercept.Z = (info.Intercept.Z - m_UpstreamZ) * idz + newupz;
                info.TopZ = (info.TopZ - m_UpstreamZ) * idz + newupz;
                info.BottomZ = (info.BottomZ - m_UpstreamZ) * idz + newupz;
                ((Segment)Segments[i]).SetInfo(info);
            }
            m_RefCenter.Z = (m_RefCenter.Z - m_UpstreamZ) * idz + newupz;
            m_UpstreamZ = newupz;
            m_DownstreamZ = newdownz;
            m_UpstreamZ_Updated = m_DownstreamZ_Updated = true;            
        }
        /// <summary>
        /// Adds one or more FlexiSegments, also setting their LayerOwner and position in the layer.
        /// </summary>
        /// <param name="seg">the new segments to be added.</param>
        /// <returns>an array with the list of the positions assigned to segments.</returns>
        /// <remarks>The array can contain null elements. In this case, the segment can be specified later.</remarks>
        public virtual int [] Add(Segment [] segs)
        {
            int n = Segments.Length;
            int dn = segs.Length;
            int[] newids = new int[dn];
            SySal.TotalScan.Segment[] newsegs = new Segment[n + dn];
            int i;
            for (i = 0; i < n; i++)
                newsegs[i] = Segments[i];
            for (i = 0; i < dn; i++)
            {
                newsegs[i + n] = segs[i];
                if (segs[i] != null) segs[i].SetLayer(this, i + n);
                newids[i] = i + n;
            }
            Segments = newsegs;
            return newids;
        }
        /// <summary>
        /// Sets a FlexiSegment to occupy a certain position in a FlexiLayer.
        /// </summary>
        /// <param name="seg">the FlexiSegment to be associated to the layer.</param>
        /// <param name="lypos">the position of the segment in the layer.</param>
        /// <remarks>The method sets the layer and position of the segment.</remarks>
        public virtual void SetSegment(Segment seg, int lypos)
        {
            seg.SetLayer(this, lypos);
            Segments[lypos] = seg;
        }
        /// <summary>
        /// Removes all segments in a layer with the specified order positions, and reclaims space. Other segments have their positions renumbered accordingly.
        /// </summary>
        /// <param name="removepos">the list of the positions to remove segments from. This list need not be ordered.</param>
        public virtual void Remove(int[] removepos)
        {
            SySal.TotalScan.Segment zs = new SySal.TotalScan.Segment(new SySal.Tracking.MIPEmulsionTrackInfo(), null);
            int dn = 0;
            foreach (int lyi in removepos)
                if (Segments[lyi] != zs)
                {
                    dn++;
                    Segments[lyi] = zs;
                }
            int n = Segments.Length;
            SySal.TotalScan.Segment[] newsegs = new Segment[n - dn];
            int i, j;            
            for (i = j = 0; i < n; i++)
                if (Segments[i] != zs)
                {
                    newsegs[j] = Segments[i];
                    if (newsegs[j] != null) ((Segment)newsegs[j]).SetLayer(this, j);
                    j++;
                }
            Segments = newsegs;    
        }
        /// <summary>
        /// Sets the alignment data for this layer, also recomputing the inverse matrix.
        /// </summary>
        /// <param name="a">the new alignment data to be used.</param>
        public void SetAlignment(AlignmentData a) { SetAlignmentData(a); }

        /// <summary>
        /// Sets the average radiation length in the layer.
        /// </summary>
        /// <param name="radlen">the value of the radiation length.</param>
        public void SetRadiationLength(double radlen) { m_RadiationLength = radlen; }

        /// <summary>
        /// Sets the average radiation length in the material downstream of the layer.
        /// </summary>
        /// <param name="radlen">the value of the radiation length.</param>
        public void SetDownstreamRadiationLength(double radlen) { m_DownstreamRadiationLength = radlen; }

        /// <summary>
        /// Sets the average radiation length in the material upstream of the layer.
        /// </summary>
        /// <param name="radlen">the value of the radiation length.</param>
        public void SetUpstreamRadiationLength(double radlen) { m_UpstreamRadiationLength = radlen; }
    }

    /// <summary>
    /// A volume track in a FlexiVolume.
    /// </summary>
    public class Track : SySal.TotalScan.Track
    {
        /// <summary>
        /// Sets the ordering number of the track in the FlexiVolume.
        /// </summary>
        /// <param name="id">the new ordering number.</param>
        public virtual void SetId(int id)
        {
            m_Id = id;
        }
        /// <summary>
        /// Property backer member for <c>DataSet</c>.
        /// </summary>
        protected DataSet m_DataSet;
        /// <summary>
        /// The data set this track belongs to.
        /// </summary>
        public virtual DataSet DataSet
        {
            get { return m_DataSet; }
            set { m_DataSet = value; }
        }
        /// <summary>
        /// Creates an empty track.
        /// </summary>
        /// <param name="ds">the dataset this track belongs to.</param>
        /// <param name="id">the id to be assigned to the track.</param>
        public Track(DataSet ds, int id)
            : base()
        {
            m_Id = id;
            m_DataSet = ds;
            Segments = new Segment[0];            
        }
        /// <summary>
        /// Adds one or more FlexiSegments, also setting their TrackOwner and position in the track.
        /// </summary>
        /// <param name="seg">the new segments to be added.</param>
        /// <returns>an array with the list of the positions assigned to segments.</returns>
        /// <remarks>The array can contain null elements. In this case, the segment can be specified later.</remarks>
        public virtual int[] AddSegments(Segment[] segs)
        {
            int n = Segments.Length;
            int dn = segs.Length;
            int[] newids = new int[dn];
            SySal.TotalScan.Segment[] newsegs = new Segment[n + dn];
            int i;
            for (i = 0; i < n; i++)
                newsegs[i] = Segments[i];
            for (i = 0; i < dn; i++)
            {
                newsegs[i + n] = segs[i];
                if (segs[i] != null) segs[i].SetTrack(this, i + n);
                newids[i] = i + n;
            }
            Segments = newsegs;
            NotifyChanged();
            return newids;
        }
        /// <summary>
        /// Sets a FlexiSegment to occupy a certain position in a FlexiTrack.
        /// </summary>
        /// <param name="seg">the FlexiSegment to be associated to the track.</param>
        /// <param name="lypos">the position of the segment in the track.</param>
        /// <remarks>The method sets the layer and position of the segment.</remarks>
        public virtual void SetSegment(Segment seg, int tkpos)
        {
            seg.SetTrack(this, tkpos);
            Segments[tkpos] = seg;
            NotifyChanged();
        }
        /// <summary>
        /// Removes all segments in a track with the specified order positions, and reclaims space. Other segments have their positions renumbered accordingly.
        /// </summary>
        /// <param name="removepos">the list of the positions to remove segments from. This list need not be ordered.</param>
        public virtual void RemoveSegments(int[] removepos)
        {
            SySal.TotalScan.Flexi.Segment zs = new SySal.TotalScan.Flexi.Segment(new SySal.TotalScan.Segment(new SySal.Tracking.MIPEmulsionTrackInfo(), new NullIndex()), null);
            int dn = 0;
            foreach (int lyi in removepos)
                if (Segments[lyi] != zs)
                {
                    dn++;
                    Segments[lyi] = zs;
                }
            int n = Segments.Length;
            SySal.TotalScan.Segment[] newsegs = new Segment[n - dn];
            int i, j;
            for (i = j = 0; i < n; i++)
                if (Segments[i] != zs)
                    ((Segment)(newsegs[j++] = Segments[i])).SetTrack(this, j);
            Segments = newsegs;
            NotifyChanged();
        }

        /// <summary>
        /// Retrieves the base tracks in this track.
        /// </summary>
        public SySal.Tracking.MIPEmulsionTrackInfo[] BaseTracks
        {
            get
            {
                int nbt = 0;
                int i;
                for (i = 0; i < Segments.Length; i++)
                    if (Segments[i].LayerOwner.Side == 0 && Segments[i].Info.Sigma >= 0.0) nbt++;
                    else if (Segments[i].LayerOwner.Side == 1 &&
                        (i + 1) < Segments.Length &&
                        Segments[i + 1].LayerOwner.Side == 2 &&
                        Segments[i].LayerOwner.SheetId == Segments[i + 1].LayerOwner.SheetId &&
                        ((SySal.TotalScan.Flexi.Segment)Segments[i]).DataSet.DataId == ((SySal.TotalScan.Flexi.Segment)Segments[i + 1]).DataSet.DataId)
                    {
                        nbt++;
                        i++;
                    }
                SySal.Tracking.MIPEmulsionTrackInfo[] ret = new SySal.Tracking.MIPEmulsionTrackInfo[nbt];
                nbt = 0;
                for (i = 0; i < Segments.Length; i++)
                    if (Segments[i].LayerOwner.Side == 0 && Segments[i].Info.Sigma >= 0.0) 
                        (ret[nbt++] = Segments[i].Info).Field = (uint)Segments[i].LayerOwner.Id;
                    else if (Segments[i].LayerOwner.Side == 1 &&
                        (i + 1) < Segments.Length &&
                        Segments[i + 1].LayerOwner.Side == 2 &&
                        Segments[i].LayerOwner.SheetId == Segments[i + 1].LayerOwner.SheetId &&
                        Segments[i].LayerOwner.BrickId == Segments[i + 1].LayerOwner.BrickId &&
                        ((SySal.TotalScan.Flexi.Segment)Segments[i]).DataSet.DataId == ((SySal.TotalScan.Flexi.Segment)Segments[i + 1]).DataSet.DataId &&
                        ((SySal.TotalScan.Flexi.Segment)Segments[i]).DataSet.DataType == ((SySal.TotalScan.Flexi.Segment)Segments[i + 1]).DataSet.DataType)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo s_top = Segments[i].Info;
                        SySal.Tracking.MIPEmulsionTrackInfo s_bottom = Segments[i + 1].Info;
                        SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                        info.Field = (uint)Segments[i].LayerOwner.SheetId;
                        info.Count = (ushort)(s_top.Count + s_bottom.Count);
                        info.AreaSum = s_top.AreaSum + s_bottom.AreaSum;
                        info.TopZ = s_top.TopZ;
                        info.BottomZ = s_bottom.BottomZ;
                        info.Intercept.Z = s_top.BottomZ;
                        info.Intercept.X = s_top.Intercept.X + (info.Intercept.Z - s_top.Intercept.Z) * s_top.Slope.X;
                        info.Intercept.Y = s_top.Intercept.Y + (info.Intercept.Z - s_top.Intercept.Z) * s_top.Slope.Y;
                        info.Slope.X = (info.Intercept.X - s_bottom.Intercept.X - (s_bottom.TopZ - s_bottom.Intercept.Z) * s_bottom.Slope.X) / (s_top.BottomZ - s_bottom.TopZ);
                        info.Slope.Y = (info.Intercept.Y - s_bottom.Intercept.Y - (s_bottom.TopZ - s_bottom.Intercept.Z) * s_bottom.Slope.Y) / (s_top.BottomZ - s_bottom.TopZ);
                        info.Slope.Z = 1.0;
                        info.Sigma = Math.Max(s_top.Sigma, s_bottom.Sigma);
                        ret[nbt++] = info;
                        i++;
                    }
                return ret;
            }
        }
        }

    /// <summary>
    /// A vertex in a FlexiVolume
    /// </summary>
    public class Vertex : SySal.TotalScan.Vertex
    {
        /// <summary>
        /// Sets the ordering number of the vertex in the FlexiVolume.
        /// </summary>
        /// <param name="id">the new ordering number.</param>
        public virtual void SetId(int id)
        {
            m_Id = id;
        }
        /// <summary>
        /// Property backer member for <c>DataSet</c>.
        /// </summary>
        protected DataSet m_DataSet;
        /// <summary>
        /// The data set this vertex belongs to.
        /// </summary>
        public virtual DataSet DataSet
        {
            get { return m_DataSet; }
            set { m_DataSet = value; }
        }
        /// <summary>
        /// Creates an empty vertex.
        /// </summary>
        /// <param name="ds">the dataset this vertex belongs to.</param>
        /// <param name="id">the id to be assigned to the vertex.</param>
        public Vertex(DataSet ds, int id)
            : base()
        {
            m_Id = id;
            m_DataSet = ds;
        }

        internal void SetPos(double x, double y, double z, double dx, double dy, double avgd)
        {
            m_X = x;
            m_Y = y;
            m_Z = z;
            m_DX = dx;
            m_DY = dy;
            m_AverageDistance = avgd;
            m_VertexCoordinatesUpdated = true;
        }
    }    

    /// <summary>
    /// A FlexiVolume, i.e. a Volume that can host data from several data sets.
    /// </summary>
    public class Volume : SySal.TotalScan.Volume
    {
        /// <summary>
        /// A list of layers in a FlexiVolume.
        /// </summary>
        public class LayerList : SySal.TotalScan.Volume.LayerList
        {
            /// <summary>
            /// Inserts a layer in Z order (using its <c>RefCenter.Z</c> information).
            /// </summary>
            /// <param name="ly">the layer to be inserted.</param>
            /// <remarks>the ids of the layers (including the newly inserted one) are recomputed.</remarks>
            public virtual void Insert(Layer ly)
            {
                int n = Items.Length;
                SySal.TotalScan.Layer[] newitems = new SySal.TotalScan.Layer[n + 1];
                int i, j;
                double z = ly.RefCenter.Z;
                for (i = 0; i < n && z < Items[i].RefCenter.Z; i++) ;
                for (j = 0; j < i; j++) newitems[j] = Items[j];
                newitems[j] = ly;
                ly.SetId(j);
                while (i < n)                
                    ((Layer)(newitems[++j] = Items[i++])).SetId(j);
                Items = newitems;
            }
            /// <summary>
            /// Removes the layer at the specified position.
            /// </summary>
            /// <param name="pos">the position hosting the layer to be removed.</param>
            /// <remarks>the ids of the layers following the removed layer are recomputed.</remarks>
            public virtual void Remove(int pos)
            {
                int n = Items.Length;
                SySal.TotalScan.Layer[] newitems = new SySal.TotalScan.Layer[n - 1];
                int i;
                for (i = 0; i < pos; i++) newitems[i] = Items[i];
                for (++i; i < n; i++) ((Layer)(newitems[i - 1] = Items[i])).SetId(i - 1);
                Items = newitems;
            }
            /// <summary>
            /// Builds an empty list.
            /// </summary>
            public LayerList() : base() { Items = new SySal.TotalScan.Layer[0]; }
        }

        /// <summary>
        /// A list of tracks in a FlexiVolume.
        /// </summary>
        public class TrackList : SySal.TotalScan.Volume.TrackList
        {
            /// <summary>
            /// Adds a list of tracks.
            /// </summary>
            /// <param name="tks">the tracks to be added. Null values are allowed, leaving empty entries that can be filled later.</param>
            /// <returns>the ids assigned to the tracks</returns>
            public virtual int [] Insert(Track [] tks)
            {
                int n = Items.Length;
                int dn = tks.Length;
                SySal.TotalScan.Track[] newitems = new SySal.TotalScan.Track[n + dn];
                int i;
                int [] newids = new int[dn];
                for (i = 0; i < n; i++) newitems[i] = Items[i];
                for (i = 0; i < dn; i++)
                {
                    newitems[i + n] = tks[i];
                    if (tks[i] != null) tks[i].SetId(i + n);
                    newids[i] = i + n;
                }
                Items = newitems;
                return newids;
            }
            /// <summary>
            /// Sets a track in the track list at the position specified by its Id property.
            /// </summary>
            /// <param name="tk">the track to be set.</param>
            public virtual void Set(Track tk)
            {
                Items[tk.Id] = tk;
            }
            /// <summary>
            /// Removes the tracks at the specified positions.
            /// </summary>
            /// <param name="pos">the position hosting the tracks to be removed.</param>
            /// <remarks>the ids of the tracks are recomputed.</remarks>
            public virtual void Remove(int [] pos)
            {
                int n = Items.Length;
                int dn = 0;
                SySal.TotalScan.Track ztk = new SySal.TotalScan.Track();
                foreach (int tki in pos)
                    if (Items[tki] != ztk)
                    {
                        Items[tki] = ztk;
                        dn++;
                    }
                SySal.TotalScan.Track[] newitems = new SySal.TotalScan.Track[n - dn];
                int i, j;
                for (i = j = 0; i < n; i++)
                    if (Items[i] != ztk)
                    {
                        newitems[j] = Items[i];
                        if (newitems[j] != null) ((Track)newitems[j]).SetId(j);                        
                        j++;
                    }
                Items = newitems;
            }
            /// <summary>
            /// Builds an empty list.
            /// </summary>
            public TrackList() : base() { Items = new SySal.TotalScan.Track[0]; }
        }

        /// <summary>
        /// A list of vertices in a FlexiVolume.
        /// </summary>
        public class VertexList : SySal.TotalScan.Volume.VertexList
        {
            /// <summary>
            /// Adds a list of vertices.
            /// </summary>
            /// <param name="tks">the vertices to be added. Null values are allowed, leaving empty entries that can be filled later.</param>
            /// <returns>the ids assigned to the vertices</returns>
            public virtual int[] Insert(Vertex[] vxs)
            {
                int n = Items.Length;
                int dn = vxs.Length;
                SySal.TotalScan.Vertex[] newitems = new SySal.TotalScan.Vertex[n + dn];
                int i;
                int[] newids = new int[dn];
                for (i = 0; i < n; i++) newitems[i] = Items[i];
                for (i = 0; i < dn; i++)
                {
                    newitems[i + n] = vxs[i];
                    if (vxs[i] != null) vxs[i].SetId(i + n);
                    newids[i] = i + n;
                }
                Items = newitems;
                return newids;
            }
            /// <summary>
            /// Sets a vertex in the vertex list at the position specified by its Id property.
            /// </summary>
            /// <param name="tk">the vertex to be set.</param>
            public virtual void Set(Vertex vx)
            {
                Items[vx.Id] = vx;
            }
            /// <summary>
            /// Removes the vertices at the specified positions.
            /// </summary>
            /// <param name="pos">the position hosting the vertex to be removed.</param>
            /// <remarks>the ids of the vertices are recomputed.</remarks>
            public virtual void Remove(int[] pos)
            {
                int n = Items.Length;
                int dn = 0;
                SySal.TotalScan.Vertex zvx = new SySal.TotalScan.Vertex();
                foreach (int tki in pos)
                    if (Items[tki] != zvx)
                    {
                        Items[tki] = zvx;
                        dn++;
                    }
                SySal.TotalScan.Vertex[] newitems = new SySal.TotalScan.Vertex[n - dn];
                int i, j;
                for (i = j = 0; i < n; i++)
                    if (Items[i] != zvx)
                    {
                        newitems[j] = Items[i];
                        if (newitems[j] != null) ((Vertex)newitems[j]).SetId(j);
                        j++;
                    }
                Items = newitems;
            }
            /// <summary>
            /// Builds an empty list.
            /// </summary>
            public VertexList() : base() { Items = new SySal.TotalScan.Vertex[0]; }
        }

        /// <summary>
        /// Builds an empty volume.
        /// </summary>
        public Volume() : base() 
        {
            m_Layers = new LayerList();
            m_Tracks = new TrackList();
            m_Vertices = new VertexList();
        }

        /// <summary>
        /// Sets the volume extents.
        /// </summary>
        /// <param name="c">the cuboid specifying the extents.</param>
        public void SetExtents(SySal.BasicTypes.Cuboid c)
        {
            m_Extents = c;
        }

        /// <summary>
        /// Sets the reference center.
        /// </summary>
        /// <param name="r">the new reference center.</param>
        public void SetRefCenter(SySal.BasicTypes.Vector r)
        {
            m_RefCenter = r;            
        }

        /// <summary>
        /// Sets the Id;
        /// </summary>
        /// <param name="id">the new id.</param>
        public void SetId(SySal.BasicTypes.Identifier id)
        {
            m_Id = id;
        }

        /// <summary>
        /// The list of the datasets that have at least one data element associated in this volume.
        /// </summary>
        public virtual DataSet[] DataSets
        {
            get
            {
                System.Collections.ArrayList dsarr = new System.Collections.ArrayList();
                int i, j;
                for (i = 0; i < m_Vertices.Length; i++)
                    if (IsIn(dsarr, ((Vertex)m_Vertices[i]).DataSet) == false)
                        dsarr.Add(((Vertex)m_Vertices[i]).DataSet);
                for (i = 0; i < m_Tracks.Length; i++)
                    if (IsIn(dsarr, ((Track)m_Tracks[i]).DataSet) == false)
                        dsarr.Add(((Track)m_Tracks[i]).DataSet);
                for (i = 0; i < m_Layers.Length; i++)
                {
                    Layer ly = (Layer)m_Layers[i];
                    for (j = 0; j < ly.Length; j++)
                        if (IsIn(dsarr, ((Segment)ly[j]).DataSet) == false)
                            dsarr.Add(((Segment)ly[j]).DataSet);
                }
                return (DataSet [])dsarr.ToArray(typeof(DataSet));
            }
        }

        public static bool IsIn(System.Collections.IList dsarr, DataSet ds)
        {
            foreach (DataSet dsx in dsarr)
                if (DataSet.AreEqual(dsx, ds)) 
                    return true;
            return false;
        }
        /// <summary>
        /// Imports a TotalScan Volume.
        /// </summary>
        /// <param name="ds">the dataset to which the volume to be imported belongs.</param>
        /// <param name="v">the volume to be imported.</param>
        public virtual void ImportVolume(DataSet ds, SySal.TotalScan.Volume v)
        {
            ImportVolume(ds, v, null);
        }
        /// <summary>
        /// Imports a TotalScan Volume.
        /// </summary>
        /// <param name="ds">the dataset to which the volume to be imported belongs.</param>
        /// <param name="v">the volume to be imported.</param>
        /// <param name="fds">the dataset that should be imported; if this parameter is <c>null</c>, all datasets are imported.</param>
        /// <remarks>The dataset filter only applies to tracks and vertices. All segments are always imported. Track/Vertex dataset consistency should be guaranteed by the user.</remarks>
        public virtual void ImportVolume(DataSet ds, SySal.TotalScan.Volume v, DataSet fds)
        {
            System.Collections.ArrayList dsa = new System.Collections.ArrayList();
            dsa.Add(ds);
            SySal.BasicTypes.Cuboid c = v.Extents;
            if (c.MinX < m_Extents.MinX) m_Extents.MinX = c.MinX;
            if (c.MaxX > m_Extents.MaxX) m_Extents.MaxX = c.MaxX;
            if (c.MinY < m_Extents.MinY) m_Extents.MinY = c.MinY;
            if (c.MaxY > m_Extents.MaxY) m_Extents.MaxY = c.MaxY;
            if (c.MinZ < m_Extents.MinZ) m_Extents.MinZ = c.MinZ;
            if (c.MaxZ > m_Extents.MaxZ) m_Extents.MaxZ = c.MaxZ;
            if (m_Layers.Length == 0) m_RefCenter = v.RefCenter;
            int i, j;
            Layer[] tl = new Layer[v.Layers.Length];
            bool[] isnewlayer = new bool[v.Layers.Length];
            int[] oldlength = new int[v.Layers.Length];
            for (i = 0; i < v.Layers.Length; i++)
            {
                for (j = 0; j < Layers.Length && (Layers[j].BrickId != v.Layers[i].BrickId || Layers[j].SheetId != v.Layers[i].SheetId || Layers[j].Side != v.Layers[i].Side); j++) ;
                if (j == Layers.Length)
                {
                    isnewlayer[i] = true;
                    tl[i] = new Layer(v.Layers[i], ds);
                    ((LayerList)m_Layers).Insert(tl[i]);
                }
                else
                {
                    isnewlayer[i] = false;                    
                    tl[i] = (SySal.TotalScan.Flexi.Layer)Layers[j];
                    oldlength[i] = tl[i].Length;
                    SySal.TotalScan.Flexi.Segment[] segs = new SySal.TotalScan.Flexi.Segment[v.Layers[i].Length];
                    SySal.TotalScan.Layer li = v.Layers[i];
                    for (j = 0; j < segs.Length; j++) segs[j] = SySal.TotalScan.Flexi.Segment.Copy(li[j], ds); //new SySal.TotalScan.Flexi.Segment(li[j], ds);
                    tl[i].Add(segs);
                }
            }

            Track[] tt = null;// = new Track[v.Tracks.Length];
            System.Collections.ArrayList ato = new System.Collections.ArrayList();
            int[] ixremap = new int[v.Tracks.Length];
            for (i = 0; i < v.Tracks.Length; i++)
            {
                SySal.TotalScan.Track otk = v.Tracks[i];
                if (otk is SySal.TotalScan.Flexi.Track)
                    if (fds != null && SySal.TotalScan.Flexi.DataSet.AreEqual(fds, ((SySal.TotalScan.Flexi.Track)otk).DataSet) == false)
                    {
                        ixremap[i] = -1;
                        continue;
                    }
                ixremap[i] = m_Tracks.Length + i;
                Track tk = new Track(ds, ixremap[i]);
                SySal.TotalScan.Flexi.DataSet tds = null;
                if (otk is SySal.TotalScan.Flexi.Track) tds = ((SySal.TotalScan.Flexi.Track)otk).DataSet;
                SySal.TotalScan.Attribute[] a = otk.ListAttributes();
                foreach (SySal.TotalScan.Attribute a1 in a)
                {
                    if (tds == null && a1.Index is SySal.TotalScan.NamedAttributeIndex && ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.StartsWith(DataSetString))
                    {
                        tds = new DataSet();
                        tds.DataType = ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.Substring(DataSetString.Length);
                        tds.DataId = (long)a1.Value;                        
                    }
                    else tk.SetAttribute(a1.Index, a1.Value);
                }
                if (fds != null && (tds == null || SySal.TotalScan.Flexi.DataSet.AreEqual(fds, tds))) tds = ds;
                if (tds != null)
                {
                    bool found = false;
                    foreach (SySal.TotalScan.Flexi.DataSet dsi in dsa)
                        if (SySal.TotalScan.Flexi.DataSet.AreEqual(dsi, tds))
                        {
                            tds = dsi;
                            found = true;
                            break;
                        }
                    if (found == false) dsa.Add(tds);
                    tk.DataSet = tds;
                }                
                SySal.TotalScan.Flexi.Segment[] segs = new SySal.TotalScan.Flexi.Segment[otk.Length];
                for (j = 0; j < segs.Length; j++)
                    if (otk[j].PosInLayer >= 0)
                    {
                        /*
                        segs[j] = (SySal.TotalScan.Flexi.Segment)v.Layers[otk[j].LayerOwner.Id][otk[j].PosInLayer];
                        segs[j].DataSet = tk.DataSet;
                         */                        
                        if (isnewlayer[otk[j].LayerOwner.Id]) segs[j] = (SySal.TotalScan.Flexi.Segment)tl[otk[j].LayerOwner.Id][otk[j].PosInLayer];
                        else segs[j] = (SySal.TotalScan.Flexi.Segment)tl[otk[j].LayerOwner.Id][oldlength[otk[j].LayerOwner.Id] + otk[j].PosInLayer];
                        segs[j].DataSet = tk.DataSet;
                    }
                    else
                    {
                        (segs[j] = SySal.TotalScan.Flexi.Segment.Copy(otk[j], tk.DataSet)).SetLayer(tl[otk[j].LayerOwner.Id], -1);
                        tl[otk[j].LayerOwner.Id].Add(new SySal.TotalScan.Flexi.Segment[1] { segs[j] } );
                        segs[j].DataSet = tk.DataSet;
                    }
                tk.AddSegments(segs);
                ato.Add(tk);
            }
            tt = (SySal.TotalScan.Flexi.Track [])ato.ToArray(typeof(SySal.TotalScan.Flexi.Track));
            ato.Clear();
            Vertex[] tv = null; // new Vertex[v.Vertices.Length];            
            for (i = 0; i < v.Vertices.Length; i++)
            {
                SySal.TotalScan.Vertex ovx = v.Vertices[i];
                if (ovx is SySal.TotalScan.Flexi.Vertex)
                    if (fds != null && SySal.TotalScan.Flexi.DataSet.AreEqual(fds, ((SySal.TotalScan.Flexi.Vertex)ovx).DataSet) == false) continue;
                Vertex vx = new Vertex(ds, m_Vertices.Length + i);                                
                SySal.TotalScan.Flexi.DataSet tds = null;
                if (ovx is SySal.TotalScan.Flexi.Vertex) tds = ((SySal.TotalScan.Flexi.Vertex)ovx).DataSet;
                SySal.TotalScan.Attribute[] a = ovx.ListAttributes();                
                foreach (SySal.TotalScan.Attribute a1 in a)
                {
                    if (tds == null && a1.Index is SySal.TotalScan.NamedAttributeIndex && ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.StartsWith(DataSetString))
                    {
                        tds = new DataSet();
                        tds.DataType = ((SySal.TotalScan.NamedAttributeIndex)a1.Index).Name.Substring(DataSetString.Length);
                        tds.DataId = (long)a1.Value;
                    }
                    else vx.SetAttribute(a1.Index, a1.Value);
                }                
                if (fds != null && (tds == null || SySal.TotalScan.Flexi.DataSet.AreEqual(fds, tds))) tds = ds;
                if (tds != null)
                {
                    bool found = false;
                    foreach (SySal.TotalScan.Flexi.DataSet dsi in dsa)
                        if (SySal.TotalScan.Flexi.DataSet.AreEqual(dsi, tds))
                        {
                            tds = dsi;
                            found = true;
                            break;
                        }
                    if (found == false) dsa.Add(tds);
                    vx.DataSet = tds;
                }
                for (j = 0; j < ovx.Length; j++)
                {                    
                    SySal.TotalScan.Track otk = ovx[j];
                    if (ixremap[otk.Id] < 0) break;
                    if (otk.Upstream_Vertex == ovx)
                    {
                        vx.AddTrack(tt[ixremap[otk.Id]], false);
                        tt[ixremap[otk.Id]].SetUpstreamVertex(vx);
                    }
                    else
                    {
                        vx.AddTrack(tt[ixremap[otk.Id]], true);
                        tt[ixremap[otk.Id]].SetDownstreamVertex(vx);
                    }
                }
                if (j < ovx.Length) continue;
                vx.SetPos(ovx.X, ovx.Y, ovx.Z, ovx.DX, ovx.DY, ovx.AverageDistance);
                ato.Add(vx);                               
            }
            tv = (SySal.TotalScan.Flexi.Vertex[])ato.ToArray(typeof(SySal.TotalScan.Flexi.Vertex));
            ato.Clear();
            ixremap = null;
            ((TrackList)m_Tracks).Insert(tt);
            ((VertexList)m_Vertices).Insert(tv);
        }
        /// <summary>
        /// Saves a TotalScan volume to a stream.
        /// </summary>
        /// <param name="w">the stream to save to.</param>
        public override void Save(System.IO.Stream w)
        {
            Type firsttype = null;
            int i, j;
            for (i = 0; i < m_Layers.Length && m_Layers[i].Length > 0; i++)
            {
                firsttype = m_Layers[i][0].Index.GetType();
                break;
            }
            for (; i < m_Layers.Length && firsttype != null; i++)
                for (j = 0; j < m_Layers[i].Length; j++)
                    if (m_Layers[i][j].Index.GetType() != firsttype)
                    {
                        firsttype = null;
                        break;
                    }
            if (firsttype == null)
                for (i = 0; i < m_Layers.Length; i++)
                    for (j = 0; j < m_Layers[i].Length; j++)
                        ((SySal.TotalScan.Flexi.Segment)m_Layers[i][j]).SetIndex(new SySal.TotalScan.NullIndex());

            for (i = 0; i < m_Tracks.Length; i++)
            {
                SySal.TotalScan.Flexi.Track tk = (SySal.TotalScan.Flexi.Track)m_Tracks[i];
                SySal.TotalScan.Flexi.DataSet ds = tk.DataSet;
                tk.SetAttribute(new SySal.TotalScan.NamedAttributeIndex(DataSetString + ds.DataType), ds.DataId);                
            }

            base.Save(w);
        }

        public const string DataSetString = "$DS$";        
    }
}
