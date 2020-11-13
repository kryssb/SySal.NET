using System;

namespace SySal.Executables.TSRClean
{
	class Volume : SySal.TotalScan.Volume
	{
		class Layer : SySal.TotalScan.Layer
		{
			public Layer(SySal.TotalScan.Layer l, int mintracksegs)
			{
				int i, j;
				for (i = j = 0; i < l.Length; i++)
					if (l[i].TrackOwner != null && (l[i].TrackOwner.Length >= mintracksegs || l[i].TrackOwner.Upstream_Vertex != null || l[i].TrackOwner.Downstream_Vertex != null)) j++;
				SySal.TotalScan.Segment [] segs = new SySal.TotalScan.Segment[j];
				for (i = j = 0; i < l.Length; i++)
					if (l[i].TrackOwner != null && (l[i].TrackOwner.Length >= mintracksegs || l[i].TrackOwner.Upstream_Vertex != null || l[i].TrackOwner.Downstream_Vertex != null)) 
						segs[j++] = l[i];
				AddSegments(segs);
				m_AlignmentData = l.AlignData;
				m_DownstreamZ = l.DownstreamZ;
				m_DownstreamZ_Updated = true;
				m_Id = l.Id;
				m_RefCenter = l.RefCenter;
                m_BrickId = l.BrickId;
				m_SheetId = l.SheetId;
                m_Side = l.Side;
				m_UpstreamZ = l.UpstreamZ;
				m_UpstreamZ_Updated = true;
			}
		}

		class LayerList : SySal.TotalScan.Volume.LayerList
		{
			public LayerList(SySal.TotalScan.Volume.LayerList ll, int mintracksegs)
			{
				Items = new Layer[ll.Length];
				int i;
				for (i = 0; i < ll.Length; i++)
					Items[i] = new Layer(ll[i], mintracksegs);
			}
		}

		class Track : SySal.TotalScan.Track
		{
			public static void mySetId(SySal.TotalScan.Track t, int id)
			{
				Track.SetId(t, id);
			}
		}

		class TrackList : SySal.TotalScan.Volume.TrackList
		{
			public TrackList(SySal.TotalScan.Volume.TrackList tl, int mintracksegs)
			{
				int i, j;
				for (i = j = 0; i < tl.Length; i++)
					if (tl[i].Length >= mintracksegs || tl[i].Upstream_Vertex != null || tl[i].Downstream_Vertex != null) j++;
				Items = new SySal.TotalScan.Track[j];
				for (i = j = 0; i < tl.Length; i++)
					if (tl[i].Length >= mintracksegs || tl[i].Upstream_Vertex != null || tl[i].Downstream_Vertex != null)
						Items[j++] = tl[i];
				for (i = 0; i < j; i++)
					Track.mySetId(Items[i], i);
			}
		}

		public Volume(SySal.TotalScan.Volume v, int mintracksegs)
		{
			m_Extents = v.Extents;
			m_Id = v.Id;
			m_RefCenter = v.RefCenter;
			m_Layers = new LayerList(v.Layers, mintracksegs);
			m_Tracks = new TrackList(v.Tracks, mintracksegs);
			m_Vertices = v.Vertices;
		}
	}

	/// <summary>
	/// TSRClean - Command line tool to clean TSR files.	
	/// </summary>
	/// <remarks>
	/// <para>The function of TSRClean is to clean TSR files of background. Only the volume tracks 
	/// with a minimum number of segments, or those attached to vertices. </para>
	/// <para>Usage: <c>TSRClean &lt;input Opera file&gt; &lt;output Opera file&gt; &lt;min track segments&gt;</c></para>
	/// </remarks>
	public class Exe
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			if (args.Length != 3)
			{
				Console.WriteLine("usage: TSRClean <input Opera file> <output Opera file> <min track segments>");
				return;
			}
            SySal.TotalScan.NullIndex.RegisterFactory();
            SySal.TotalScan.BaseTrackIndex.RegisterFactory();
            SySal.TotalScan.MIPMicroTrackIndex.RegisterFactory();
            SySal.TotalScan.NamedAttributeIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex.RegisterFactory();
            SySal.OperaDb.TotalScan.DBNamedAttributeIndex.RegisterFactory();            
			SySal.TotalScan.Volume v = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(args[0], typeof(SySal.TotalScan.Volume));
			Volume sv = new Volume(v, Convert.ToInt32(args[2]));
			SySal.OperaPersistence.Persist(args[1], sv);
		}
	}
}
