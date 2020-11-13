using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Executables.EasyReconstruct
{
    /// <summary>
    /// Stores information on the preferences of the current user.
    /// </summary>
    [Serializable]
    public class UserProfileInfo
    {
        /// <summary>
        /// The list of preferred map merging filters.
        /// </summary>
        public string[] MapMergeFilters = new string[0];
        /// <summary>
        /// The list of preferred track/segement filters.
        /// </summary>
        public string[] DisplayFilters = new string[0];

        private static System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(UserProfileInfo));
        /// <summary>
        /// The instance bound to this process.
        /// </summary>
        public static UserProfileInfo ThisProfileInfo = new UserProfileInfo();

        private const string FileName = "EasyReconstruct.profile";

        /// <summary>
        /// Saves the current instance to the user profile file.
        /// </summary>
        public static void Save()
        {
            string p = System.Environment.GetFolderPath(System.Environment.SpecialFolder.ApplicationData);
            if (p.EndsWith("/") == false && p.EndsWith("\\") == false) p += "/";
            p += FileName;
            System.IO.StringWriter sw = new System.IO.StringWriter();
            xmls.Serialize(sw, ThisProfileInfo);
            System.IO.File.WriteAllText(p, sw.ToString());
        }

        /// <summary>
        /// Load user preferences from the user profile file.
        /// </summary>
        public static void Load()
        {
            try
            {
                string p = System.Environment.GetFolderPath(System.Environment.SpecialFolder.ApplicationData);
                if (p.EndsWith("/") == false && p.EndsWith("\\") == false) p += "/";
                p += FileName;
                System.IO.StringReader sr = new System.IO.StringReader(System.IO.File.ReadAllText(p));
                ThisProfileInfo = (UserProfileInfo)xmls.Deserialize(sr);
            }
            catch (Exception x)
            {                
                ThisProfileInfo.MapMergeFilters = new string[]
                    {
                        "N >= 20 && sqrt(sx^2+sy^2)>0.03 && sqrt(sx^2+sy^2)<0.4", 
                        "NT >= 4 && sqrt(sx^2+sy^2)>0.03 && sqrt(sx^2+sy^2)<0.4"
                    };
                ThisProfileInfo.DisplayFilters = new string[]
                    {
                        "N >= 20 || (S < 0 && N >= 12)",
                        "(N >= 20 || (S < 0 && N >= 12)) && SQRT(SX^2+SY^2)>0.01",
                        "G/N > 18 * SQRT(1 + USX^2 + USY^2)",
                        "G/N > 19 * SQRT(1 + USX^2 + USY^2)",
                        "G/N > 20 * SQRT(1 + USX^2 + USY^2)",
                        "G/N > 18 * SQRT(1 + USX^2 + USY^2) && SQRT(USX^2 + USY^2) > 0.01",
                        "G/N > 19 * SQRT(1 + USX^2 + USY^2) && SQRT(USX^2 + USY^2) > 0.01",
                        "G/N > 20 * SQRT(1 + USX^2 + USY^2) && SQRT(USX^2 + USY^2) > 0.01"
                    };
            }
        }
    }
}

