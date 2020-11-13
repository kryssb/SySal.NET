using System;
using System.Xml.Serialization;

namespace SySal.Executables.BatchLink
{
	/// <summary>
	/// Batch link configuration.
	/// </summary>
	[Serializable]
	[XmlType("BatchLink.Config")]
	public class Config
	{
		private double _topMultSlopeX;
		private double _topMultSlopeY;
		private double _bottomMultSlopeX;
		private double _bottomMultSlopeY;
		private double _topDeltaSlopeX;
		private double _topDeltaSlopeY;
		private double _bottomDeltaSlopeX;
		private double _bottomDeltaSlopeY;
		private double _maskBinning;
		private double _maskPeakHeightMultiplier;
		private bool _autoCorrectMultipliers;
		private double _autoCorrectMinSlope;
		private double _autoCorrectMaxSlope;
		public SySal.Processing.StripesFragLink2.Configuration LinkerConfig;

		public double TopMultSlopeX { get{return _topMultSlopeX;} set{_topMultSlopeX = value;} }
		public double TopMultSlopeY { get{return _topMultSlopeY;} set{_topMultSlopeY = value;} }
		public double BottomMultSlopeX { get{return _bottomMultSlopeX;} set{_bottomMultSlopeX = value;} }  
		public double BottomMultSlopeY { get{return _bottomMultSlopeY;} set{_bottomMultSlopeY = value;} }  
		public double TopDeltaSlopeX { get{return _topDeltaSlopeX;} set{_topDeltaSlopeX = value;} }  
		public double TopDeltaSlopeY { get{return _topDeltaSlopeY;} set{_topDeltaSlopeY = value;} } 
		public double BottomDeltaSlopeX { get{return _bottomDeltaSlopeX;} set{_bottomDeltaSlopeX = value;} } 
		public double BottomDeltaSlopeY { get{return _bottomDeltaSlopeY;} set{_bottomDeltaSlopeY = value;} } 
		public double MaskBinning { get{return _maskBinning;} set{_maskBinning = value;} }
		public double MaskPeakHeightMultiplier { get{return _maskPeakHeightMultiplier;} set{_maskPeakHeightMultiplier = value;} }
		public bool AutoCorrectMultipliers { get{return _autoCorrectMultipliers;} set{_autoCorrectMultipliers = value;} }
		public double AutoCorrectMinSlope { get{return _autoCorrectMinSlope;} set{_autoCorrectMinSlope = value;} }
		public double AutoCorrectMaxSlope { get{return _autoCorrectMaxSlope;} set{_autoCorrectMaxSlope = value;} }
		public int MinGrains { get{if (LinkerConfig == null) return 0; else return LinkerConfig.MinGrains;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.MinGrains = value;} }
		public double MinSlope { get{if (LinkerConfig == null) return 0; else return LinkerConfig.MinSlope;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.MinSlope = value;} }
		public double MergePosTol { get{if (LinkerConfig == null) return 0; else return LinkerConfig.MergePosTol;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.MergePosTol = value;} }
		public double MergeSlopeTol { get{if (LinkerConfig == null) return 0; else return LinkerConfig.MergeSlopeTol;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.MergeSlopeTol = value;} }
		public double SlopeTol { get{if (LinkerConfig == null) return 0; else return LinkerConfig.SlopeTol;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.SlopeTol = value;} }
		public double SlopeTolIncreaseWithSlope { get{if (LinkerConfig == null) return 0; else return LinkerConfig.SlopeTolIncreaseWithSlope;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.SlopeTolIncreaseWithSlope = value;} }
		public uint MemorySaving { get{if (LinkerConfig == null) return 0; else return LinkerConfig.MemorySaving;} set {if (LinkerConfig == null) LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration(); LinkerConfig.MemorySaving = value;}}
		
		public string ToXml()
		{
			XmlSerializer xmls = new XmlSerializer(this.GetType());
			System.IO.StringWriter strw = new System.IO.StringWriter();
			xmls.Serialize(strw, this);			
			return strw.ToString();
		}
	}
}
