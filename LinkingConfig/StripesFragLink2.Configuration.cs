using System;
using System.Xml.Serialization;

namespace SySal.Processing.StripesFragLink2
{
	/// <summary>
	/// Configuration for StripesFragmentLinker.
	/// </summary>
	[Serializable]
	[XmlType("StripesFragLink2.Configuration")]
	public class Configuration : SySal.Management.Configuration
	{
		public int MinGrains;
		public double MinSlope;
		public double MergePosTol;
		public double MergeSlopeTol;
		public double PosTol;
		public double SlopeTol;
		public double SlopeTolIncreaseWithSlope;
		public uint MemorySaving;

		public Configuration() : base("") {}

		public Configuration(string name) : base(name) {}

		public override object Clone()
		{
			Configuration c = new Configuration(Name);
			c.MinGrains = MinGrains;
			c.MinSlope = MinSlope;
			c.MergePosTol = MergePosTol;
			c.MergeSlopeTol = MergeSlopeTol;
			c.PosTol = PosTol;
			c.SlopeTol = SlopeTol;
			c.SlopeTolIncreaseWithSlope = SlopeTolIncreaseWithSlope;
			c.MemorySaving = MemorySaving;
			return c;
		}
	}
}
