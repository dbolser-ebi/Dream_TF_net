package no.uib.bccs.jaspar.JasparClient;

import org.apache.axis2.transport.http.HTTPConstants;

import no.uib.bccs.jaspar.JasparDBStub;
import no.uib.bccs.jaspar.JasparDBStub.Database_type0;
import no.uib.bccs.jaspar.JasparDBStub.Format_type0;
import no.uib.bccs.jaspar.JasparDBStub.GetAllMatrices;
import no.uib.bccs.jaspar.JasparDBStub.GetAllMatricesResponse;
import no.uib.bccs.jaspar.JasparDBStub.MatrixType;
import no.uib.bccs.jaspar.JasparDBStub.TagType;


public class JasparClient {

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception{

        JasparDBStub stub = new JasparDBStub();

        stub._getServiceClient().getOptions().setProperty(HTTPConstants.CHUNKED, false);

        GetAllMatrices request = new JasparDBStub.GetAllMatrices();
        request.setDatabase(Database_type0.CORE);
        request.setFormat(Format_type0.PFM);

        GetAllMatricesResponse response = stub.getAllMatrices(request);
        MatrixType[] matrices = response.getMatrix();

        for (MatrixType matrix : matrices) {
            System.out.println("Matrix: " + matrix.getID());
            System.out.print("A ");
            for (String i : matrix.getA().getCol()) {
                System.out.print(i + " ");
            }
            System.out.print("\nT ");
            for (String i : matrix.getT().getCol()) {
                System.out.print(i + " ");
            }
            System.out.print("\nC ");
            for (String i : matrix.getC().getCol()) {
                System.out.print(i + " ");
            }
            System.out.print("\nG ");
            for (String i : matrix.getG().getCol()) {
                System.out.print(i + " ");
            }

            System.out.println("\n\nTags:");
            TagType[] tags = matrix.getTag();
            for (TagType tag : tags) {
                for (String value : tag.getValue()) {
                    System.out.println(tag.getName() + ": " + value);
                }
            }
            System.out.println();

        }



    }

}