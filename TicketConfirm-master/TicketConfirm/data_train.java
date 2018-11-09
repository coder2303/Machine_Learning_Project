import java.io.*;
import java.util.*;
public class data_train {
    public static String getD()
    {
        String date="";

        date=(1 + (int)(Math.random() * 29))+"";
        return date;
    }
    public static String getM()
    {
        String date="";

        date=""+(1 + (int)(Math.random() * 12));
        return date;
    }
    public static String getY()
    {
        String date="";
        int yearBegin=2013;
        int yearEnd=2017-yearBegin;

        date=""+(yearBegin + (int)(Math.random() * yearEnd));
        return date;
    }
    static	HashMap<String,Integer> map=new HashMap();
    static String station_codes[] ;
    static void STN_CODES(){
     for(int i=0;i<713;i++){
         map.put(station_codes[i],i+1);
     }
    }

    static int FIND_CODE(String stn){
        return   map.get(stn);
    }
    public static void main(String[]args) throws Exception{
        String splitBy = ",";
        BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\Aman Joshi\\Desktop\\pandey proj\\final new\\sttncodes.csv"));
        String line=null;
        station_codes=new String[713];
        int lmn=0;
        while((line = br.readLine()) != null){
            String[] b = line.split(splitBy);
            station_codes[lmn]=b[0];
           lmn++;
        }

        for(int i=0;i<station_codes.length;i++){
            station_codes[i]=station_codes[i].toUpperCase();
        }
        Arrays.sort(station_codes);
        //System.out.println(Arrays.toString(station_codes));
        br.close();
        PrintWriter pw = new PrintWriter(new File("train.csv"));
        StringBuilder sb = new StringBuilder();
        Random rand = new Random();
        STN_CODES();
        sb.append("TRAIN NO.");
        sb.append(',');
        sb.append("CLASS");
        sb.append(',');
        sb.append("FROM");
        sb.append(',');
        sb.append("FROM_CODE");
        sb.append(',');
        sb.append("TO");
        sb.append(',');
        sb.append("TO_CODE");
        sb.append(',');
        sb.append("JOURNEY-DAY");
        sb.append(',');
        sb.append("JOURNEY-MONTH");
        sb.append(',');
        sb.append("JOURNEY-YEAR");
        sb.append(',');
        sb.append("QUOTA");
        sb.append(',');
        sb.append("BOOKING-DAY");
        sb.append(',');
        sb.append("BOOKING-MONTH");
        sb.append(',');
        sb.append("BOOKING-YEAR");
        sb.append(',');
        sb.append("WAITING NO.");

        sb.append(',');
        sb.append("CONFIRMED");
        sb.append('\n');
        for(int i=0;i<10000;i++) {
            int rand1 = rand.nextInt(1000)+10000;
            sb.append(rand1);
            sb.append(',');
            rand1 = rand.nextInt(7)+1;
            sb.append(rand1);
            sb.append(',');
            rand1=rand.nextInt(712)+1;
            String output = station_codes[rand1];
            sb.append(output);
            sb.append(',');
            int val=FIND_CODE(output);
            sb.append(val);
            sb.append(',');
            rand1=rand.nextInt(712)+1;
            output =  station_codes[rand1];
            sb.append(output);

            sb.append(',');
            val=FIND_CODE(output);
            sb.append(val);
            sb.append(',');


            sb.append(getD());
            sb.append(',');
            sb.append(getM());
            sb.append(',');
            sb.append(getY());
            sb.append(',');
            rand1 = rand.nextInt(11)+1;
            sb.append(rand1);
            sb.append(',');
            sb.append(getD());
            sb.append(',');
            sb.append(getM());
            sb.append(',');
            sb.append(getY());
            sb.append(',');
            int rand2 = rand.nextInt(140)+1;
            sb.append(rand2);
            sb.append(',');
            rand1 = rand.nextInt(2);
            if(rand2>80)
                rand1=0;
            sb.append(rand1);

            sb.append('\n');
        }
        pw.write(sb.toString());
        pw.close();
        System.out.println("done!");
    }
}