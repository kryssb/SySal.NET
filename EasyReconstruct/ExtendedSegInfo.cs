using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.EasyReconstruct
{
    /// <summary>
    /// Extended information stored in a segment.
    /// </summary>
    public interface IExtendedSegmentInformation
    {
        /// <summary>
        /// The list of additional fields.
        /// </summary>
        string[] ExtendedFields { get; }
        /// <summary>
        /// Returns the value of the field with the specified name.
        /// </summary>
        /// <param name="name">the name of the field to retrieve.</param>
        /// <returns>the value of the field.</returns>
        object ExtendedField(string name);
        /// <summary>
        /// Returns the type of the field with the specified name.
        /// </summary>
        /// <param name="name">the name of the field.</param>
        /// <returns>the type of the field.</returns>
        Type ExtendedFieldType(string name);
    }
    /// <summary>
    /// Allows accessing additional information stored in segments.
    /// </summary>
    public class ExtendedSegInfoProvider : IExtendedSegmentInformation
    {
        /// <summary>
        /// The segment from which data have to be extracted.
        /// </summary>
        protected SySal.TotalScan.Segment m_Segment;
        /// <summary>
        /// Sets the segment from which data have to be extracted.
        /// </summary>
        public SySal.TotalScan.Segment Segment
        {            
            get
            {
                return m_Segment;
            }
            set
            {
                m_Segment = value;
            }
        }

        string[] m_Fields = new string[0];
        object[] m_Defaults = new object[0];

        /// <summary>
        /// Sets the list of extended fields to extract.
        /// </summary>
        /// <param name="fields">the list of fields to extract.</param>
        /// <param name="defaults">the list of default values.</param>
        public void SetExtendedFieldsList(string[] fields, object [] defaults)
        {
            if (fields.Length != defaults.Length) throw new Exception("The number of default values must be equal to the number of fields.");
            m_Fields = fields;
            m_Defaults = defaults;
        }

        #region IExtendedSegmentInformation Members

        /// <summary>
        /// The set of extended fields.
        /// </summary>
        public string[] ExtendedFields
        {
            get { return m_Fields; }
        }

        /// <summary>
        /// Returns the value of the field with the specified name.
        /// </summary>
        /// <param name="name">the name of the field to retrieve.</param>
        /// <returns>the value of the field.</returns>
        public object ExtendedField(string name)
        {
            try
            {
                return (m_Segment as IExtendedSegmentInformation).ExtendedField(name);
            }
            catch (Exception)
            {
                int i;
                for (i = 0; i < m_Fields.Length && String.Compare(m_Fields[i], name, true) != 0; i++);
                if (i == m_Fields.Length) throw new Exception("Unsupported field name \"" + name + "\".");
                return m_Defaults[i];
            }
        }
        /// <summary>
        /// Returns the type of the field with the specified name.
        /// </summary>
        /// <param name="name">the name of the field.</param>
        /// <returns>the type of the field.</returns>
        public Type ExtendedFieldType(string name)
        {
            int i;
            for (i = 0; i < m_Fields.Length && String.Compare(m_Fields[i], name, true) != 0; i++) ;
            if (i == m_Fields.Length) throw new Exception("Unsupported field name \"" + name + "\".");
            return m_Defaults[i].GetType();
        }

        #endregion
    }
}
